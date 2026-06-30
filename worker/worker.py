"""
Syn — Worker

Plukker meldinger fra Azure Service Bus-køen, henter filen fra Blob Storage,
kjører Azure Document Intelligence + dokumentid-segmentering + claim-ekstraksjon
+ embedding, og skriver alt til PostgreSQL.

Designet for å kjøre som Azure Container Apps Job, trigget av kø-meldinger.
Kan også kjøres lokalt/på Railway som en kontinuerlig polling-loop for testing.

Bruker ÉN delt databasetilkobling gjennom hele process_job() — ikke en ny
tilkobling per insert. Unngår portutmattelse ved store filer med mange sider.
"""

import os, re, json, tempfile, time, traceback
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
from openai import OpenAI
from azure.storage.blob import BlobServiceClient
from azure.servicebus import ServiceBusClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# ── Konfigurasjon ─────────────────────────────────────────────────────────────

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
AZURE_DI_ENDPOINT = os.environ["AZURE_DI_ENDPOINT"]
AZURE_DI_KEY = os.environ["AZURE_DI_KEY"]
AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
AZURE_SERVICEBUS_CONNECTION_STRING = os.environ["AZURE_SERVICEBUS_CONNECTION_STRING"]

BLOB_CONTAINER = "dokumenter"
SERVICEBUS_QUEUE = "dokument-prosessering"
EMBED_MODEL = "text-embedding-3-large"
MAX_WORDS_FIN = 400
MAX_WORDS_GROV = 1500
MIN_WORDS = 40

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def utcnow():
    return datetime.now(timezone.utc)


# ── PostgreSQL — én delt tilkobling per jobb ──────────────────────────────────

def db():
    """Brukes kun for engangskall utenfor process_job (statusoppdateringer)."""
    return psycopg2.connect(DATABASE_URL)


def update_job(job_id: str, status: str = None, progress_pct: int = None,
               current_step: str = None, error_message: str = None,
               started: bool = False, completed: bool = False):
    """Egen kortvarig tilkobling er OK her — kalles sjelden, ikke i tett løkke."""
    conn = db()
    try:
        with conn.cursor() as cur:
            sets, params = [], []
            if status is not None:
                sets.append("status = %s"); params.append(status)
            if progress_pct is not None:
                sets.append("progress_pct = %s"); params.append(progress_pct)
            if current_step is not None:
                sets.append("current_step = %s"); params.append(current_step)
            if error_message is not None:
                sets.append("error_message = %s"); params.append(error_message)
            if started:
                sets.append("started_at = %s"); params.append(utcnow())
            if completed:
                sets.append("completed_at = %s"); params.append(utcnow())
            params.append(job_id)
            cur.execute(f"UPDATE processing_jobs SET {', '.join(sets)} WHERE job_id = %s", params)
            conn.commit()
    finally:
        conn.close()


def start_processing_run(conn, run_type: str, model_name: str, prompt_version: str = "") -> str:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO processing_runs (run_type, model_name, prompt_version, status)
               VALUES (%s, %s, %s, 'running') RETURNING run_id""",
            (run_type, model_name, prompt_version)
        )
        run_id = cur.fetchone()[0]
        conn.commit()
        return str(run_id)


def complete_processing_run(conn, run_id: str, status: str = "completed", notes: str = ""):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE processing_runs SET completed_at = %s, status = %s, notes = %s WHERE run_id = %s",
            (utcnow(), status, notes, run_id)
        )
        conn.commit()


def insert_page(conn, file_id: str, page_number: int, ocr_text: str,
                 sub_dokumentid: Optional[str], azure_run_id: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO pages (file_id, page_number, ocr_text, sub_dokumentid, azure_di_run_id)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (file_id, page_number) DO UPDATE
               SET ocr_text = EXCLUDED.ocr_text, sub_dokumentid = EXCLUDED.sub_dokumentid
               RETURNING page_id""",
            (file_id, page_number, ocr_text, sub_dokumentid, azure_run_id)
        )
        page_id = cur.fetchone()[0]
        conn.commit()
        return str(page_id)


def insert_claim(conn, page_id: str, run_id: str, speaker: str, speaker_role: str,
                  claim_text: str, normalized_claim: str, claim_type: str,
                  topic: str, event_date: Optional[str], confidence: float):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO claims
               (page_id, run_id, speaker, speaker_role, claim_text, normalized_claim,
                claim_type, topic, event_date, extraction_confidence)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (page_id, run_id, speaker, speaker_role, claim_text, normalized_claim,
             claim_type, topic, event_date, confidence)
        )
        conn.commit()


def insert_chunk(conn, case_id: str, file_id: str, page_id_start: str, page_id_slutt: str,
                  text: str, granularitet: str, side_start: int, side_slutt: int,
                  sub_dokumentid: str, doc_type: str, person: str, dato: str,
                  embedding: list, embedding_run_id: str):
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO chunks
               (case_id, file_id, page_id_start, page_id_slutt, text, granularitet,
                side_start, side_slutt, sub_dokumentid, doc_type, person, dato,
                kildetype, embedding, embedding_run_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'korpus', %s::vector, %s)""",
            (case_id, file_id, page_id_start, page_id_slutt, text, granularitet,
             side_start, side_slutt, sub_dokumentid, doc_type, person, dato,
             vec_str, embedding_run_id)
        )
        conn.commit()


def get_file_metadata(conn, file_id: str) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM source_files WHERE file_id = %s", (file_id,))
        row = cur.fetchone()
        return dict(row) if row else {}


def get_previous_part_last_paragraph(conn, pdf_id: str, del_nummer: int) -> str:
    """Cross-file stitching: henter siste avsnitt fra forrige dels siste fine chunk."""
    if del_nummer <= 1:
        return ""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT c.text FROM chunks c
            JOIN source_files sf ON sf.file_id = c.file_id
            WHERE sf.pdf_id = %s AND sf.del_nummer = %s AND c.granularitet = 'fin'
            ORDER BY c.side_slutt DESC LIMIT 1
        """, (pdf_id, del_nummer - 1))
        row = cur.fetchone()
        if not row:
            return ""
        avsnitt = [p for p in row["text"].split("\n\n") if p.strip()]
        return avsnitt[-1] if avsnitt else ""


# ── Dokumentid-deteksjon ───────────────────────────────────────────────────────

def decode_doubled(text: str) -> str:
    result = []
    i = 0
    while i < len(text) - 1:
        if text[i] == text[i + 1] and text[i].isalnum():
            result.append(text[i]); i += 2
        else:
            result.append(text[i]); i += 1
    if i < len(text):
        result.append(text[i])
    return ''.join(result)


def extract_footer_dokumentid(footer_text: str) -> Optional[str]:
    decoded = decode_doubled(footer_text)
    match = re.search(r'dokumentid[:\s]*(\d+)', decoded, re.IGNORECASE)
    return match.group(1) if match else None


# ── Azure Document Intelligence ───────────────────────────────────────────────

def extract_with_azure(pdf_path: str) -> list[dict]:
    client = DocumentIntelligenceClient(AZURE_DI_ENDPOINT, AzureKeyCredential(AZURE_DI_KEY))
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-read", analyze_request=f, content_type="application/octet-stream"
        )
    result = poller.result()

    pages = []
    for page in result.pages:
        lines = [line.content for line in (page.lines or [])]
        if not lines:
            continue
        footer_text = " ".join(lines[-3:])
        pages.append({
            "page_num": page.page_number,
            "text": "\n".join(lines),
            "dokumentid": extract_footer_dokumentid(footer_text),
        })
    return pages


# ── Segmentering ──────────────────────────────────────────────────────────────

def segment_into_subdocs(pages: list[dict]) -> list[dict]:
    subdocs = []
    current_id, current_pages = None, []
    for page in pages:
        pid = page["dokumentid"]
        if pid and pid != current_id:
            if current_pages:
                subdocs.append({"dokumentid": current_id or "ukjent", "sider": current_pages})
            current_id = pid
            current_pages = [page]
        else:
            current_pages.append(page)
    if current_pages:
        subdocs.append({"dokumentid": current_id or "ukjent", "sider": current_pages})
    return subdocs


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_into_paragraphs(text: str) -> list[str]:
    raw = re.split(r'\n{2,}|\n(?=[A-ZÆØÅ0-9])', text)
    return [p.strip() for p in raw if len(p.split()) >= 5]


def build_chunks(pages: list[dict], max_words: int, granularitet: str,
                  prefix_context: str = "") -> list[dict]:
    all_paras = []
    for page in pages:
        for p in split_into_paragraphs(page["text"]):
            all_paras.append({"text": p, "page": page["page_num"]})
    if not all_paras:
        return []

    chunks = []
    i = 0
    last_para = {"text": prefix_context, "page": all_paras[0]["page"]} if prefix_context else None

    while i < len(all_paras):
        chunk_paras, chunk_pages, word_count = [], set(), 0
        if last_para:
            chunk_paras.append(last_para["text"])
            chunk_pages.add(last_para["page"])
            word_count += len(last_para["text"].split())

        while i < len(all_paras):
            para = all_paras[i]
            pw = len(para["text"].split())
            if pw > max_words:
                for sent in re.split(r'(?<=[.!?])\s+', para["text"]):
                    sw = len(sent.split())
                    if word_count + sw > max_words and word_count >= MIN_WORDS:
                        break
                    chunk_paras.append(sent); chunk_pages.add(para["page"]); word_count += sw
                i += 1
                break
            if word_count + pw > max_words and word_count >= MIN_WORDS:
                break
            chunk_paras.append(para["text"]); chunk_pages.add(para["page"]); word_count += pw
            i += 1

        if not chunk_paras:
            i += 1
            continue

        sorted_pages = sorted(chunk_pages)
        chunks.append({
            "text": "\n\n".join(chunk_paras),
            "side_start": sorted_pages[0],
            "side_slutt": sorted_pages[-1],
            "granularitet": granularitet,
        })
        last_para = all_paras[i - 1] if 0 < i <= len(all_paras) else None

    return chunks


# ── Claim-ekstraksjon ──────────────────────────────────────────────────────────

CLAIM_PROMPT = """Du analyserer en side fra et norsk rettsdokument. Identifiser konkrete \
påstander (claims) i teksten — utsagn der noen hevder noe om en hendelse, observasjon eller fakta.

For hver påstand, returner et JSON-objekt med feltene:
- speaker: hvem som hevder dette
- speaker_role: en av "vitne", "tiltalt", "politi", "sakkyndig", "domstol", "ukjent"
- claim_text: direkte utdrag fra teksten
- normalized_claim: kort omformulering i tredjeperson
- claim_type: en av "observasjon", "forklaring", "påstand", "innrømmelse", "benektelse"
- topic: kort tema-tag
- event_date: dato i format YYYY-MM-DD, eller null
- extraction_confidence: 0.0-1.0

Returner KUN en JSON-liste, ingen annen tekst. Tom liste [] hvis ingen klare påstander.

SIDETEKST:
{text}"""


def extract_claims_from_page(page_text: str) -> list[dict]:
    if len(page_text.split()) < 15:
        return []
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": CLAIM_PROMPT.format(text=page_text)}],
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("claims", [])
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def embed(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


# ── Blob Storage ──────────────────────────────────────────────────────────────

def download_from_blob(blob_path: str) -> bytes:
    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service.get_blob_client(container=BLOB_CONTAINER, blob=blob_path)
    return blob_client.download_blob().readall()


# ── Hovedprosessering ────────────────────────────────────────────────────────

def process_job(job_payload: dict):
    job_id = job_payload["job_id"]
    file_id = job_payload["file_id"]
    case_id = job_payload["case_id"]
    blob_path = job_payload["blob_path"]

    print(f"[{job_id}] Starter prosessering: {blob_path}")
    update_job(job_id, status="processing", current_step="henter_fil", progress_pct=5, started=True)

    # ÉN delt tilkobling for hele jobben
    conn = psycopg2.connect(DATABASE_URL)

    try:
        file_meta = get_file_metadata(conn, file_id)
        pdf_id = file_meta.get("pdf_id", "")
        del_nummer = file_meta.get("del_nummer", 1)
        doc_type = file_meta.get("doc_type", "ukjent")
        person = file_meta.get("person", "")
        dato = file_meta.get("dato", "")

        # 1. Last ned fra Blob
        content = download_from_blob(blob_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # 2. Azure DI OCR
        update_job(job_id, current_step="ocr", progress_pct=15)
        azure_run_id = start_processing_run(conn, "ocr", "azure-di-prebuilt-read")
        try:
            pages = extract_with_azure(tmp_path)
            complete_processing_run(conn, azure_run_id, "completed", f"{len(pages)} sider")
        except Exception as e:
            complete_processing_run(conn, azure_run_id, "failed", str(e))
            raise
        finally:
            os.unlink(tmp_path)

        print(f"[{job_id}] OCR ferdig: {len(pages)} sider")
        update_job(job_id, current_step="lagrer_sider", progress_pct=30)

        # 3. Skriv sider til PostgreSQL
        pg_page_ids = {}
        for page in pages:
            pg_page_ids[page["page_num"]] = insert_page(
                conn, file_id, page["page_num"], page["text"], page["dokumentid"], azure_run_id
            )

        # 4. Claim-ekstraksjon
        update_job(job_id, current_step="claim_ekstraksjon", progress_pct=45)
        claim_run_id = start_processing_run(conn, "claim_extraction", "gpt-4o-mini", "v1")
        claims_count = 0
        for idx, page in enumerate(pages):
            page_claims = extract_claims_from_page(page["text"])
            pg_page_id = pg_page_ids.get(page["page_num"])
            for c in page_claims:
                try:
                    insert_claim(
                        conn, pg_page_id, claim_run_id,
                        c.get("speaker", ""), c.get("speaker_role", "ukjent"),
                        c.get("claim_text", ""), c.get("normalized_claim", ""),
                        c.get("claim_type", ""), c.get("topic", ""),
                        c.get("event_date"), c.get("extraction_confidence", 0.5)
                    )
                    claims_count += 1
                except Exception:
                    continue
            if idx % 20 == 0:
                progress = 45 + int(20 * idx / max(len(pages), 1))
                update_job(job_id, progress_pct=min(progress, 65))
        complete_processing_run(conn, claim_run_id, "completed", f"{claims_count} claims")
        print(f"[{job_id}] Claims ferdig: {claims_count}")

        # 5. Cross-file stitching + segmentering + chunking + embedding
        update_job(job_id, current_step="chunking_embedding", progress_pct=70)
        prefix_context = (
            get_previous_part_last_paragraph(conn, pdf_id, del_nummer)
            if pdf_id and del_nummer > 1 else ""
        )
        subdocs = segment_into_subdocs(pages)

        embed_run_id = start_processing_run(conn, "embedding", EMBED_MODEL)
        chunks_total = 0

        for sub_idx, subdoc in enumerate(subdocs):
            ctx = prefix_context if sub_idx == 0 else ""
            fine_chunks = build_chunks(subdoc["sider"], MAX_WORDS_FIN, "fin", ctx)
            grov_chunks = build_chunks(subdoc["sider"], MAX_WORDS_GROV, "grov", ctx)

            for chunk in fine_chunks + grov_chunks:
                if len(chunk["text"].strip()) < 50:
                    continue
                page_id_start = pg_page_ids.get(chunk["side_start"])
                page_id_slutt = pg_page_ids.get(chunk["side_slutt"])
                vector = embed(chunk["text"])
                insert_chunk(
                    conn, case_id, file_id, page_id_start, page_id_slutt,
                    chunk["text"], chunk["granularitet"],
                    chunk["side_start"], chunk["side_slutt"],
                    subdoc["dokumentid"], doc_type, person, dato,
                    vector, embed_run_id
                )
                chunks_total += 1

        complete_processing_run(conn, embed_run_id, "completed", f"{chunks_total} chunks")
        print(f"[{job_id}] Embedding ferdig: {chunks_total} chunks")

        update_job(job_id, status="completed", progress_pct=100,
                   current_step="ferdig", completed=True)
        print(f"[{job_id}] FERDIG")

    except Exception as e:
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{job_id}] FEIL: {error_detail}")
        update_job(job_id, status="failed", error_message=str(e)[:1000], completed=True)

    finally:
        conn.close()


# ── Service Bus-lytter ──────────────────────────────────────────────────────────

def run_worker_loop():
    """
    Kontinuerlig polling-loop. Egnet for Container Apps Job med min/max
    execution count, eller for testing på Railway/lokalt.
    """
    print("Syn worker starter. Lytter på kø:", SERVICEBUS_QUEUE)
    sb_client = ServiceBusClient.from_connection_string(AZURE_SERVICEBUS_CONNECTION_STRING)

    with sb_client:
        receiver = sb_client.get_queue_receiver(queue_name=SERVICEBUS_QUEUE, max_wait_time=30)
        with receiver:
            while True:
                messages = receiver.receive_messages(max_message_count=1, max_wait_time=30)
                if not messages:
                    print("Ingen jobber i kø, venter...")
                    continue
                for msg in messages:
                    try:
                        payload = json.loads(str(msg))
                        process_job(payload)
                        receiver.complete_message(msg)
                    except Exception as e:
                        print(f"Feil ved behandling av melding: {e}")
                        receiver.abandon_message(msg)


def run_worker_once():
    """
    Kjør én jobb og avslutt — riktig modus for Azure Container Apps Job,
    som forventer at containeren termineres etter fullført arbeid.
    """
    sb_client = ServiceBusClient.from_connection_string(AZURE_SERVICEBUS_CONNECTION_STRING)
    with sb_client:
        receiver = sb_client.get_queue_receiver(queue_name=SERVICEBUS_QUEUE, max_wait_time=10)
        with receiver:
            messages = receiver.receive_messages(max_message_count=1, max_wait_time=10)
            if not messages:
                print("Ingen jobber i kø.")
                return
            msg = messages[0]
            try:
                payload = json.loads(str(msg))
                process_job(payload)
                receiver.complete_message(msg)
            except Exception as e:
                print(f"Feil ved behandling av melding: {e}")
                receiver.abandon_message(msg)


if __name__ == "__main__":
    mode = os.environ.get("WORKER_MODE", "loop")
    if mode == "once":
        run_worker_once()
    else:
        run_worker_loop()
