"""
Syn — Worker v3 (Ingest-only)

Pipeline: OCR → sider til PostgreSQL → chunking → embedding → chunks til PostgreSQL.
Claim-ekstraksjon er fjernet fra ingest og kjøres som separat jobb senere.

Én enkelt databasetilkobling per jobb, åpnet én gang og sendt inn til alle funksjoner.
Azure DI henter fil direkte fra Blob Storage via SAS-URL (støtter store filer).
"""

import os, re, json, tempfile, traceback
from datetime import datetime, timezone, timedelta
from typing import Optional
import socket
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
from openai import OpenAI
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.servicebus import ServiceBusClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# ── Konfigurasjon ─────────────────────────────────────────────────────────────

DATABASE_URL                       = os.environ["DATABASE_URL"]
OPENAI_API_KEY                     = os.environ["OPENAI_API_KEY"]
AZURE_DI_ENDPOINT                  = os.environ["AZURE_DI_ENDPOINT"]
AZURE_DI_KEY                       = os.environ["AZURE_DI_KEY"]
AZURE_STORAGE_CONNECTION_STRING    = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
AZURE_SERVICEBUS_CONNECTION_STRING = os.environ["AZURE_SERVICEBUS_CONNECTION_STRING"]

BLOB_CONTAINER   = "dokumenter"
SERVICEBUS_QUEUE = "dokument-prosessering"
EMBED_MODEL      = "text-embedding-3-large"
MAX_WORDS_FIN    = 400
MAX_WORDS_GROV   = 1500
MIN_WORDS        = 40

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def utcnow():
    return datetime.now(timezone.utc)


# ── Database ──────────────────────────────────────────────────────────────────

def open_db():
    """Åpner én PostgreSQL-tilkobling med IP-oppløsning (omgår Windows TIME_WAIT)."""
    parsed = urlparse(DATABASE_URL)
    ip = socket.gethostbyname(parsed.hostname)
    return psycopg2.connect(
        host=ip,
        port=parsed.port or 5432,
        dbname=parsed.path.lstrip('/'),
        user=parsed.username,
        password=parsed.password,
        sslmode="require",
        connect_timeout=15,
    )


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


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


# ── Hovedprosessering ─────────────────────────────────────────────────────────

def process_job(job_payload: dict):
    job_id    = job_payload["job_id"]
    file_id   = job_payload["file_id"]
    case_id   = job_payload["case_id"]
    blob_path = job_payload["blob_path"]

    print(f"[{job_id}] Starter: {blob_path}")

    conn = open_db()
    print(f"[{job_id}] DB OK")

    try:
        update_job(conn, job_id, status="processing", current_step="ocr",
                   progress_pct=5, started=True)

        file_meta  = get_file_metadata(conn, file_id)
        pdf_id     = file_meta.get("pdf_id", "")
        del_nummer = file_meta.get("del_nummer", 1)
        doc_type   = file_meta.get("doc_type", "ukjent")
        person     = file_meta.get("person", "")
        dato       = file_meta.get("dato", "")

        # 1. Azure DI OCR via SAS-URL
        azure_run_id = start_run(conn, "ocr", "azure-di-prebuilt-read")
        try:
            pages = extract_with_azure(blob_path)
            complete_run(conn, azure_run_id, "completed", f"{len(pages)} sider")
        except Exception as e:
            complete_run(conn, azure_run_id, "failed", str(e))
            raise

        print(f"[{job_id}] OCR ferdig: {len(pages)} sider")
        update_job(conn, job_id, current_step="lagrer_sider", progress_pct=30)

        # 2. Sider til PostgreSQL
        pg_page_ids = {}
        for page in pages:
            pg_page_ids[page["page_num"]] = insert_page(
                conn, file_id, page["page_num"], page["text"],
                page["dokumentid"], azure_run_id
            )
        print(f"[{job_id}] Sider lagret: {len(pg_page_ids)}")

        # 3. Chunking + embedding
        update_job(conn, job_id, current_step="embedding", progress_pct=50)
        prefix_context = (
            get_previous_part_last_paragraph(conn, pdf_id, del_nummer)
            if pdf_id and del_nummer > 1 else ""
        )
        subdocs = segment_into_subdocs(pages)
        embed_run_id = start_run(conn, "embedding", EMBED_MODEL)
        chunks_total = 0

        for sub_idx, subdoc in enumerate(subdocs):
            ctx = prefix_context if sub_idx == 0 else ""
            all_chunks = (
                build_chunks(subdoc["sider"], MAX_WORDS_FIN, "fin", ctx) +
                build_chunks(subdoc["sider"], MAX_WORDS_GROV, "grov", ctx)
            )
            for chunk in all_chunks:
                if len(chunk["text"].strip()) < 50:
                    continue
                insert_chunk(
                    conn, case_id, file_id,
                    pg_page_ids.get(chunk["side_start"]),
                    pg_page_ids.get(chunk["side_slutt"]),
                    chunk["text"], chunk["granularitet"],
                    chunk["side_start"], chunk["side_slutt"],
                    subdoc["dokumentid"], doc_type, person, dato,
                    embed(chunk["text"]), embed_run_id
                )
                chunks_total += 1

            progress = 50 + int(45 * (sub_idx + 1) / max(len(subdocs), 1))
            update_job(conn, job_id, progress_pct=min(progress, 95))

        complete_run(conn, embed_run_id, "completed", f"{chunks_total} chunks")
        print(f"[{job_id}] Embedding ferdig: {chunks_total} chunks")

        update_job(conn, job_id, status="completed", progress_pct=100,
                   current_step="ferdig", completed=True)
        print(f"[{job_id}] FERDIG ✓")

    except Exception as e:
        print(f"[{job_id}] FEIL: {e}")
        update_job(conn, job_id, status="failed",
                   error_message=str(e)[:1000], completed=True)

    finally:
        conn.close()


# ── Service Bus ───────────────────────────────────────────────────────────────

def run_worker_once():
    sb = ServiceBusClient.from_connection_string(AZURE_SERVICEBUS_CONNECTION_STRING)
    with sb:
        receiver = sb.get_queue_receiver(queue_name=SERVICEBUS_QUEUE, max_wait_time=10)
        with receiver:
            messages = receiver.receive_messages(max_message_count=1, max_wait_time=10)
            if not messages:
                print("Ingen jobber i kø.")
                return
            msg = messages[0]
            try:
                process_job(json.loads(str(msg)))
                receiver.complete_message(msg)
            except Exception as e:
                print(f"Feil: {e}")
                receiver.abandon_message(msg)


def run_worker_loop():
    print("Syn worker starter. Kø:", SERVICEBUS_QUEUE)
    sb = ServiceBusClient.from_connection_string(AZURE_SERVICEBUS_CONNECTION_STRING)
    with sb:
        receiver = sb.get_queue_receiver(queue_name=SERVICEBUS_QUEUE, max_wait_time=30)
        with receiver:
            while True:
                messages = receiver.receive_messages(max_message_count=1, max_wait_time=30)
                if not messages:
                    print("Venter...")
                    continue
                msg = messages[0]
                try:
                    process_job(json.loads(str(msg)))
                    receiver.complete_message(msg)
                except Exception as e:
                    print(f"Feil: {e}")
                    receiver.abandon_message(msg)


if __name__ == "__main__":
    mode = os.environ.get("WORKER_MODE", "loop")
    if mode == "once":
        run_worker_once()
    else:
        run_worker_loop()
