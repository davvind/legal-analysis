"""
Syn — Worker v4 (Ingest-only, herdet)

Pipeline: OCR -> sider (batch) -> chunking -> batch-embedding -> atomisk chunk-swap.

Endringer mot v3:
  * Batch-embedding med retry/backoff (ikke lenger én OpenAI-kall per chunk).
  * Idempotent reingest: chunks skrives med chunk_hash + partiell unik indeks.
    Reingest av samme file_id gjøres som ATOMISK swap i én transaksjon:
    gamle aktive chunks settes is_active=FALSE, nye bulk-insertes, ett commit.
    Søk ser aldri en halvferdig blanding.
  * Ingen langlivet idle DB-tilkobling under treg embedding (Azure-gateway
    dropper idle connections). Korte tilkoblinger per fase i stedet.
  * process_job returnerer bool. Ved feil ABANDONes Service Bus-meldingen
    (retry -> dead-letter etter MaxDeliveryCount), fullføres ALDRI ved feil.
  * Windows TIME_WAIT-hacket (DNS->IP) er fjernet. På Linux kobler vi via
    hostname direkte. sslmode=require ligger i DATABASE_URL.

FORUTSETNING: migrasjon_002 (korrigert) er kjørt. chunks har chunk_hash (NOT NULL),
chunk_index, chunking_version, word_count, char_count, is_active, superseded_at,
samt uq_chunks_active_identity.

MERK cross-part stitching: get_previous_part_last_paragraph antar at del N-1 er
ferdig når del N kjører. Service Bus garanterer IKKE rekkefølge, og parallelle
replikaer garanterer det motsatte. Kjør ÉN del ende til ende først (der er dette
uproblematisk). Før multi-part parallell kjøring: serialiser per pdf_id med
Service Bus sessions (session-id = pdf_id), ELLER lagre siste avsnitt per del ved
OCR-tid og slå det opp rekkefølge-uavhengig. Ikke skaler til alle deler før det.
"""

import os, re, json, time, hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azure.servicebus import ServiceBusClient, AutoLockRenewer
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# ── Konfigurasjon ─────────────────────────────────────────────────────────────

DATABASE_URL                       = os.environ["DATABASE_URL"]
OPENAI_API_KEY                     = os.environ["OPENAI_API_KEY"]
AZURE_DI_ENDPOINT                  = os.environ["AZURE_DI_ENDPOINT"]
AZURE_DI_KEY                       = os.environ["AZURE_DI_KEY"]
AZURE_STORAGE_CONNECTION_STRING    = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
AZURE_SERVICEBUS_CONNECTION_STRING = os.environ["AZURE_SERVICEBUS_CONNECTION_STRING"]

BLOB_CONTAINER    = "dokumenter"
SERVICEBUS_QUEUE  = "dokument-prosessering"
EMBED_MODEL       = "text-embedding-3-large"   # 3072 dim — indekseres via halfvec-cast
CHUNKING_VERSION  = "chunk_v1"
MAX_WORDS_FIN     = 400
MAX_WORDS_GROV    = 1500
MIN_WORDS         = 40
MIN_CHUNK_CHARS   = 50

# Låsefornyelse: hold Service Bus-meldingslåsen i live under lange jobber
# (OCR + embedding på store deler tar mer enn standard 5 min lås). Bør matche
# Container Apps --replica-timeout.
MAX_JOB_SECONDS   = 3600

# Batch-embedding: batch avsluttes ved item-tak ELLER token-budsjett (est.),
# det som treffer først. text-embedding-3-large: 8191 tokens/input (våre chunks
# er godt under), og romslig totalbudsjett per request. Konservative tall her.
EMBED_BATCH_MAX_ITEMS  = 96
EMBED_BATCH_MAX_TOKENS = 200_000

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def utcnow():
    return datetime.now(timezone.utc)


# ── Database (korte tilkoblinger, hostname) ───────────────────────────────────

def open_db():
    # Linux: koble via hostname direkte. sslmode=require ligger i URL-en.
    return psycopg2.connect(DATABASE_URL, connect_timeout=15)


def update_job(conn, job_id: str, status: str = None, progress_pct: int = None,
               current_step: str = None, error_message: str = None,
               started: bool = False, completed: bool = False):
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
    if not sets:
        return
    params.append(job_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE processing_jobs SET {', '.join(sets)} WHERE job_id = %s", params)
    conn.commit()


def progress(job_id: str, pct: int = None, step: str = None):
    """Kort, selvstendig progress-oppdatering (egen tilkobling, ingen idle)."""
    conn = open_db()
    try:
        update_job(conn, job_id, progress_pct=pct, current_step=step)
    finally:
        conn.close()


def mark_failed(job_id: str, message: str):
    """Best-effort: marker jobb feilet. Egen tilkobling, svelger egne feil."""
    try:
        conn = open_db()
        try:
            update_job(conn, job_id, status="failed",
                       error_message=str(message)[:1000], completed=True)
        finally:
            conn.close()
    except Exception as e:
        print(f"[{job_id}] Klarte ikke markere failed: {e}")


def start_run(conn, run_type: str, model_name: str, prompt_version: str = "") -> str:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO processing_runs (run_type, model_name, prompt_version, status)
               VALUES (%s, %s, %s, 'running') RETURNING run_id""",
            (run_type, model_name, prompt_version)
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return str(run_id)


def complete_run(conn, run_id: str, status: str = "completed", notes: str = ""):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE processing_runs SET completed_at = %s, status = %s, notes = %s WHERE run_id = %s",
            (utcnow(), status, notes, run_id)
        )
    conn.commit()


def insert_pages_bulk(conn, file_id: str, pages: list[dict], azure_run_id: str) -> dict:
    """Én round-trip for alle sider. Returnerer {page_number: page_id}."""
    rows = [
        (file_id, p["page_num"], p["text"], p["dokumentid"], azure_run_id)
        for p in pages
    ]
    sql = """
        INSERT INTO pages (file_id, page_number, ocr_text, sub_dokumentid, azure_di_run_id)
        VALUES %s
        ON CONFLICT (file_id, page_number) DO UPDATE
            SET ocr_text = EXCLUDED.ocr_text,
                sub_dokumentid = EXCLUDED.sub_dokumentid
        RETURNING page_id, page_number
    """
    with conn.cursor() as cur:
        returned = execute_values(cur, sql, rows,
                                  template="(%s,%s,%s,%s,%s)", fetch=True)
    conn.commit()
    return {page_number: str(page_id) for (page_id, page_number) in returned}


def get_file_metadata(conn, file_id: str) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM source_files WHERE file_id = %s", (file_id,))
        row = cur.fetchone()
        return dict(row) if row else {}


def get_previous_part_last_paragraph(conn, pdf_id: str, del_nummer: int) -> str:
    # Se modul-docstring om rekkefølge-antagelsen. Returnerer "" hvis forrige
    # del ikke er ferdig ennå — stitching hoppes da stille over.
    if not pdf_id or del_nummer <= 1:
        return ""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT c.text FROM chunks c
            JOIN source_files sf ON sf.file_id = c.file_id
            WHERE sf.pdf_id = %s AND sf.del_nummer = %s
              AND c.granularitet = 'fin' AND c.is_active = TRUE
            ORDER BY c.side_slutt DESC LIMIT 1
        """, (pdf_id, del_nummer - 1))
        row = cur.fetchone()
        if not row:
            return ""
        avsnitt = [p for p in row["text"].split("\n\n") if p.strip()]
        return avsnitt[-1] if avsnitt else ""


def supersede_and_insert_chunks(conn, file_id: str, chunk_rows: list[tuple]) -> int:
    """
    ATOMISK reingest for én fil:
      1) gamle aktive chunks -> is_active=FALSE, superseded_at=now()
      2) bulk-insert nye chunks (ON CONFLICT DO NOTHING mot partiell unik indeks)
    Alt i én transaksjon. Ved unntak rulles alt tilbake -> ingen halvferdig tilstand.
    """
    if not chunk_rows:
        # Ingen nye chunks: la evt. gamle staa uroert (ikke slett noe stille).
        return 0

    insert_sql = """
        INSERT INTO chunks
            (case_id, file_id, page_id_start, page_id_slutt, text, granularitet,
             side_start, side_slutt, sub_dokumentid, doc_type, person, dato,
             kildetype, embedding, embedding_run_id,
             chunk_index, chunk_hash, chunking_version, word_count, char_count)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    template = ("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                "'korpus',%s::vector,%s,%s,%s,%s,%s,%s)")

    with conn:  # transaksjon: commit ved suksess, rollback ved unntak
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chunks SET is_active = FALSE, superseded_at = now() "
                "WHERE file_id = %s AND is_active = TRUE",
                (file_id,)
            )
            execute_values(cur, insert_sql, chunk_rows,
                           template=template, page_size=100)
    return len(chunk_rows)


# ── Azure Blob — SAS-URL ──────────────────────────────────────────────────────

def get_blob_sas_url(blob_path: str) -> str:
    parts = dict(p.split('=', 1) for p in AZURE_STORAGE_CONNECTION_STRING.split(';') if '=' in p)
    account_name = parts["AccountName"]
    account_key  = parts["AccountKey"]
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=BLOB_CONTAINER,
        blob_name=blob_path,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=utcnow() + timedelta(hours=2),
    )
    return f"https://{account_name}.blob.core.windows.net/{BLOB_CONTAINER}/{blob_path}?{sas_token}"


# ── Azure Document Intelligence ───────────────────────────────────────────────

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


def extract_with_azure(blob_path: str) -> list[dict]:
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    client = DocumentIntelligenceClient(AZURE_DI_ENDPOINT, AzureKeyCredential(AZURE_DI_KEY))
    url = get_blob_sas_url(blob_path)
    poller = client.begin_analyze_document(
        "prebuilt-read",
        body=AnalyzeDocumentRequest(url_source=url),
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
                    chunk_paras.append(sent)
                    chunk_pages.add(para["page"])
                    word_count += sw
                i += 1
                break
            if word_count + pw > max_words and word_count >= MIN_WORDS:
                break
            chunk_paras.append(para["text"])
            chunk_pages.add(para["page"])
            word_count += pw
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


# ── Embedding (batch + backoff) ───────────────────────────────────────────────

def _est_tokens(text: str) -> int:
    return len(text) // 4 + 1


def _batches(texts: list[str]):
    batch, tok = [], 0
    for t in texts:
        est = _est_tokens(t)
        if batch and (len(batch) >= EMBED_BATCH_MAX_ITEMS or tok + est > EMBED_BATCH_MAX_TOKENS):
            yield batch
            batch, tok = [], 0
        batch.append(t)
        tok += est
    if batch:
        yield batch


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embedder en liste tekster i ett kall, med eksponentiell backoff."""
    last_err = None
    for attempt in range(6):
        try:
            resp = openai_client.embeddings.create(input=texts, model=EMBED_MODEL)
            ordered = sorted(resp.data, key=lambda d: d.index)
            return [d.embedding for d in ordered]
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"  embedding-batch feilet (forsøk {attempt+1}): {type(e).__name__} — venter {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Embedding feilet etter retries: {last_err}")


def embed_all(records: list[dict], job_id: str):
    """Embedder records['text'] in-place -> records['embedding']. Logger token/rate."""
    texts = [r["text"] for r in records]
    vectors: list[list[float]] = []
    done = 0
    total = len(texts)
    for batch in _batches(texts):
        vectors.extend(embed_batch(batch))
        done += len(batch)
        # Grov fremdrift i embedding-fasen: 50 -> 90 %.
        pct = 50 + int(40 * done / max(total, 1))
        progress(job_id, pct=min(pct, 90), step="embedding")
        print(f"[{job_id}] embeddet {done}/{total}")
    for r, v in zip(records, vectors):
        r["embedding"] = v


def vec_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Hovedprosessering ─────────────────────────────────────────────────────────

def process_job(job_payload: dict) -> bool:
    """
    Returnerer True ved full suksess, False ved feil.
    Kalleren fullfører meldingen KUN ved True; ved False/unntak abandones den
    (Service Bus retryer opptil MaxDeliveryCount og dead-letter'er deretter).
    """
    job_id    = job_payload["job_id"]
    file_id   = job_payload["file_id"]
    case_id   = job_payload["case_id"]
    blob_path = job_payload["blob_path"]

    print(f"[{job_id}] Starter: {blob_path}")

    try:
        # ── Fase A: OCR + sider (egen tilkobling, lukkes før embedding) ──────
        conn = open_db()
        try:
            update_job(conn, job_id, status="processing", current_step="ocr",
                       progress_pct=5, started=True)

            file_meta  = get_file_metadata(conn, file_id)
            pdf_id     = file_meta.get("pdf_id", "")
            del_nummer = file_meta.get("del_nummer", 1)
            doc_type   = file_meta.get("doc_type", "ukjent")
            person     = file_meta.get("person", "")
            dato       = file_meta.get("dato") or None

            azure_run_id = start_run(conn, "ocr", "azure-di-prebuilt-read")
            try:
                pages = extract_with_azure(blob_path)
                complete_run(conn, azure_run_id, "completed", f"{len(pages)} sider")
            except Exception as e:
                complete_run(conn, azure_run_id, "failed", str(e))
                raise

            print(f"[{job_id}] OCR ferdig: {len(pages)} sider")

            if not pages:
                update_job(conn, job_id, status="completed", progress_pct=100,
                           current_step="ferdig_tom", completed=True)
                print(f"[{job_id}] Ingen sider — ferdig (tom).")
                return True

            update_job(conn, job_id, current_step="lagrer_sider", progress_pct=30)
            pg_page_ids = insert_pages_bulk(conn, file_id, pages, azure_run_id)
            print(f"[{job_id}] Sider lagret: {len(pg_page_ids)}")

            prefix_context = get_previous_part_last_paragraph(conn, pdf_id, del_nummer)
            embed_run_id   = start_run(conn, "embedding", EMBED_MODEL)
        finally:
            conn.close()

        # ── Fase B: bygg chunks (ren Python, ingen DB) ──────────────────────
        subdocs = segment_into_subdocs(pages)
        records = []
        idx = 0
        for sub_idx, subdoc in enumerate(subdocs):
            ctx = prefix_context if sub_idx == 0 else ""
            for gran, maxw in (("fin", MAX_WORDS_FIN), ("grov", MAX_WORDS_GROV)):
                for ch in build_chunks(subdoc["sider"], maxw, gran, ctx):
                    txt = ch["text"].strip()
                    if len(txt) < MIN_CHUNK_CHARS:
                        continue
                    records.append({
                        "text": txt,
                        "granularitet": ch["granularitet"],
                        "side_start": ch["side_start"],
                        "side_slutt": ch["side_slutt"],
                        "sub_dokumentid": subdoc["dokumentid"],
                        "chunk_index": idx,
                    })
                    idx += 1

        print(f"[{job_id}] Bygde {len(records)} chunks — starter embedding")

        # ── Fase C: batch-embedding (treg, ingen DB-tilkobling holdes åpen) ──
        if records:
            embed_all(records, job_id)

        # ── Fase D: atomisk chunk-swap (ny tilkobling, én transaksjon) ──────
        chunk_rows = []
        for r in records:
            chunk_rows.append((
                case_id, file_id,
                pg_page_ids.get(r["side_start"]),
                pg_page_ids.get(r["side_slutt"]),
                r["text"], r["granularitet"],
                r["side_start"], r["side_slutt"],
                r["sub_dokumentid"], doc_type, person, dato,
                vec_literal(r["embedding"]), embed_run_id,
                r["chunk_index"], _sha256(r["text"]), CHUNKING_VERSION,
                len(r["text"].split()), len(r["text"]),
            ))

        conn = open_db()
        try:
            inserted = supersede_and_insert_chunks(conn, file_id, chunk_rows)
            complete_run(conn, embed_run_id, "completed", f"{inserted} chunks")
            update_job(conn, job_id, status="completed", progress_pct=100,
                       current_step="ferdig", completed=True)
        finally:
            conn.close()

        print(f"[{job_id}] FERDIG ✓  ({inserted} chunks)")
        return True

    except Exception as e:
        print(f"[{job_id}] FEIL: {e}")
        mark_failed(job_id, e)
        return False


# ── Service Bus ───────────────────────────────────────────────────────────────

def _handle_message(receiver, msg, renewer):
    """Fullfør KUN ved suksess; ellers abandon -> retry/dead-letter.
    Låsen fornyes automatisk under prosessering slik at den ikke utløper på
    lange jobber (og meldingen ikke redeleveres/dobbeltprosesseres unødvendig)."""
    renewer.register(receiver, msg, max_lock_renewal_duration=MAX_JOB_SECONDS)
    try:
        ok = process_job(json.loads(str(msg)))
    except Exception as e:
        print(f"Uventet feil i process_job: {e}")
        ok = False
    if ok:
        receiver.complete_message(msg)
    else:
        receiver.abandon_message(msg)


def run_worker_once():
    sb = ServiceBusClient.from_connection_string(AZURE_SERVICEBUS_CONNECTION_STRING)
    renewer = AutoLockRenewer(max_lock_renewal_duration=MAX_JOB_SECONDS)
    with sb, renewer:
        receiver = sb.get_queue_receiver(queue_name=SERVICEBUS_QUEUE, max_wait_time=10)
        with receiver:
            messages = receiver.receive_messages(max_message_count=1, max_wait_time=10)
            if not messages:
                print("Ingen jobber i kø.")
                return
            _handle_message(receiver, messages[0], renewer)


def run_worker_loop():
    print("Syn worker starter. Kø:", SERVICEBUS_QUEUE)
    sb = ServiceBusClient.from_connection_string(AZURE_SERVICEBUS_CONNECTION_STRING)
    renewer = AutoLockRenewer(max_lock_renewal_duration=MAX_JOB_SECONDS)
    with sb, renewer:
        receiver = sb.get_queue_receiver(queue_name=SERVICEBUS_QUEUE, max_wait_time=30)
        with receiver:
            while True:
                messages = receiver.receive_messages(max_message_count=1, max_wait_time=30)
                if not messages:
                    print("Venter...")
                    continue
                _handle_message(receiver, messages[0], renewer)


if __name__ == "__main__":
    mode = os.environ.get("WORKER_MODE", "loop")
    if mode == "once":
        run_worker_once()
    else:
        run_worker_loop()
