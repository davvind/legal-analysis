from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os, hashlib, json, uuid
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
import psycopg2.extras
from pydantic import BaseModel
from typing import List, Optional
from azure.storage.blob import BlobServiceClient
from azure.servicebus import ServiceBusClient, ServiceBusMessage

load_dotenv()

app = FastAPI(title="Syn API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["*"],
    max_age=3600,
)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DEFAULT_CASE_NAME = "orderud"
BLOB_CONTAINER = "dokumenter"
SERVICEBUS_QUEUE = "file-processing"
EMBED_DIM = 3072


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def db():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def ensure_default_case() -> str:
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT case_id FROM cases WHERE case_name = %s", (DEFAULT_CASE_NAME,))
            row = cur.fetchone()
            if row:
                return str(row[0])
            cur.execute(
                "INSERT INTO cases (case_name, description) VALUES (%s, %s) RETURNING case_id",
                (DEFAULT_CASE_NAME, "Standardsak, opprettet automatisk")
            )
            case_id = cur.fetchone()[0]
            conn.commit()
            return str(case_id)
    finally:
        conn.close()


def insert_source_file_pending(case_id: str, filename: str, checksum: str, pdf_id: str,
                                del_nummer: int, file_size: int, doc_type: str,
                                person: str, dato: str, blob_path: str) -> str:
    """
    Inserter source_file med blob_path satt, men UTEN sider/claims ennå —
    de fylles av worker-prosessen. Returnerer file_id (eksisterende hvis duplikat).
    """
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT file_id FROM source_files WHERE sha256_checksum = %s", (checksum,))
            existing = cur.fetchone()
            if existing:
                return str(existing[0])

            cur.execute(
                """INSERT INTO source_files
                   (case_id, original_filename, sha256_checksum, pdf_id, del_nummer,
                    file_size_bytes, doc_type, person, dato, blob_path)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING file_id""",
                (case_id, filename, checksum, pdf_id, del_nummer, file_size,
                 doc_type, person, dato, blob_path)
            )
            file_id = cur.fetchone()[0]
            conn.commit()
            return str(file_id)
    finally:
        conn.close()


def insert_processing_job(case_id: str, file_id: str, job_type: str, blob_path: str) -> str:
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO processing_jobs
                   (case_id, file_id, job_type, status, blob_path)
                   VALUES (%s, %s, %s, 'queued', %s) RETURNING job_id""",
                (case_id, file_id, job_type, blob_path)
            )
            job_id = cur.fetchone()[0]
            conn.commit()
            return str(job_id)
    finally:
        conn.close()


def get_job_status(job_id: str) -> dict:
    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT job_id, status, progress_pct, current_step, error_message,
                       queued_at, started_at, completed_at
                FROM processing_jobs WHERE job_id = %s
            """, (job_id,))
            row = cur.fetchone()
            return dict(row) if row else {}
    finally:
        conn.close()


# ── Azure Blob Storage ──────────────────────────────────────────────────────

def upload_to_blob(content: bytes, blob_path: str) -> None:
    blob_service = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    )
    blob_client = blob_service.get_blob_client(container=BLOB_CONTAINER, blob=blob_path)
    blob_client.upload_blob(content, overwrite=True)


# ── Azure Service Bus ────────────────────────────────────────────────────────

def send_job_message(job_id: str, file_id: str, case_id: str, blob_path: str,
                      job_type: str = "full_ingest") -> None:
    sb_client = ServiceBusClient.from_connection_string(
        os.environ["AZURE_SERVICEBUS_CONNECTION_STRING"]
    )
    payload = {
        "job_id": job_id,
        "file_id": file_id,
        "case_id": case_id,
        "blob_path": blob_path,
        "job_type": job_type,
    }
    with sb_client:
        sender = sb_client.get_queue_sender(queue_name=SERVICEBUS_QUEUE)
        with sender:
            message = ServiceBusMessage(json.dumps(payload))
            sender.send_messages(message)


# ── Embedding (for søk, ikke for opplasting) ─────────────────────────────────

def embed(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding


def vector_literal(embedding: list[float]) -> str:
    """Konverterer Python-liste til pgvector sin tekstrepresentasjon for SQL."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


# ── Endepunkter ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Syn API kjører"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/health/db")
def health_db():
    try:
        conn = db()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}


@app.post("/upload")
async def upload_korpus(
    file: UploadFile = File(...),
    doc_type: str = Form("ukjent"),
    person: str = Form(""),
    dato: str = Form(""),
    pdf_id: str = Form(""),
    del_nummer: int = Form(1),
):
    """
    Asynkron opplasting. Laster filen til Blob Storage, oppretter en
    processing_job, legger melding i kø, og returnerer UMIDDELBART —
    venter ikke på OCR eller claim-ekstraksjon.
    """
    case_id = ensure_default_case()

    content = await file.read()
    checksum = hashlib.sha256(content).hexdigest()
    file_size = len(content)

    # Blob-sti: case/pdf_id/del_nummer_filnavn for ryddig organisering
    blob_path = f"{DEFAULT_CASE_NAME}/{pdf_id or 'ukjent'}/del{del_nummer}_{file.filename}"

    # Last opp råfil til Blob Storage FØR noe annet — dette er rask filoverføring,
    # ingen prosessering, håndterer flere GB uten timeout
    upload_to_blob(content, blob_path)

    # Skriv source_file-rad (idempotent via sjekksum)
    file_id = insert_source_file_pending(
        case_id, file.filename, checksum, pdf_id or file.filename,
        del_nummer, file_size, doc_type, person, dato, blob_path
    )

    # Opprett jobb og legg i kø
    job_id = insert_processing_job(case_id, file_id, "full_ingest", blob_path)
    send_job_message(job_id, file_id, case_id, blob_path)

    return {
        "status": "queued",
        "job_id": job_id,
        "file_id": file_id,
        "case_id": case_id,
        "filename": file.filename,
        "blob_path": blob_path,
        "message": "Filen er lastet opp og lagt i kø for prosessering. Bruk /upload/status/{job_id} for å følge fremdrift."
    }


@app.get("/upload/status/{job_id}")
def upload_status(job_id: str):
    """Frontend poller dette endepunktet for å vise fremdrift."""
    status = get_job_status(job_id)
    if not status:
        return {"error": "Fant ingen jobb med denne IDen"}
    return status


@app.get("/upload/status")
def upload_status_for_case(case_id: str = "", limit: int = 20):
    """Lister nylige jobber, eventuelt filtrert på sak."""
    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = """
                SELECT j.job_id, j.status, j.progress_pct, j.current_step,
                       j.queued_at, j.started_at, j.completed_at,
                       sf.original_filename
                FROM processing_jobs j
                LEFT JOIN source_files sf ON sf.file_id = j.file_id
                WHERE 1=1
            """
            params = []
            if case_id:
                query += " AND j.case_id = %s"
                params.append(case_id)
            query += " ORDER BY j.queued_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            return {"jobs": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()


@app.get("/claims")
def list_claims(speaker: str = "", review_status: str = "", limit: int = 50):
    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = """
                SELECT c.claim_id, c.speaker, c.speaker_role, c.claim_text,
                       c.normalized_claim, c.claim_type, c.topic, c.event_date,
                       c.extraction_confidence, c.review_status,
                       p.page_number, sf.original_filename
                FROM claims c
                JOIN pages p ON p.page_id = c.page_id
                JOIN source_files sf ON sf.file_id = p.file_id
                WHERE 1=1
            """
            params = []
            if speaker:
                query += " AND c.speaker ILIKE %s"
                params.append(f"%{speaker}%")
            if review_status:
                query += " AND c.review_status = %s"
                params.append(review_status)
            query += " ORDER BY c.created_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            rows = cur.fetchall()
            return {"claims": [dict(r) for r in rows], "count": len(rows)}
    finally:
        conn.close()


@app.get("/claims/sammenlign")
def sammenlign_claims_mot_kilde(filnavn: str, side: int):
    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT p.page_id, p.page_number, p.ocr_text, p.sub_dokumentid,
                       sf.original_filename, sf.pdf_id, sf.del_nummer
                FROM pages p
                JOIN source_files sf ON sf.file_id = p.file_id
                WHERE sf.original_filename = %s AND p.page_number = %s
                ORDER BY sf.upload_time DESC
                LIMIT 1
            """, (filnavn, side))
            page_row = cur.fetchone()
            if not page_row:
                return {"error": f"Fant ingen side {side} i fil '{filnavn}'"}
            page = dict(page_row)

            cur.execute("""
                SELECT claim_id, speaker, speaker_role, claim_text,
                       normalized_claim, claim_type, topic, event_date,
                       extraction_confidence, review_status, review_comment
                FROM claims WHERE page_id = %s ORDER BY created_at
            """, (page["page_id"],))
            claims = [dict(r) for r in cur.fetchall()]

            return {
                "fil": page["original_filename"],
                "side": page["page_number"],
                "sub_dokumentid": page["sub_dokumentid"],
                "ocr_tekst": page["ocr_text"],
                "antall_claims": len(claims),
                "claims": claims,
            }
    finally:
        conn.close()


@app.get("/claims/fil-oversikt")
def fil_oversikt(filnavn: str):
    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT p.page_number, p.sub_dokumentid,
                       COUNT(c.claim_id) AS antall_claims,
                       AVG(c.extraction_confidence) AS snitt_confidence
                FROM pages p
                JOIN source_files sf ON sf.file_id = p.file_id
                LEFT JOIN claims c ON c.page_id = p.page_id
                WHERE sf.original_filename = %s
                GROUP BY p.page_number, p.sub_dokumentid
                ORDER BY p.page_number
            """, (filnavn,))
            rows = [dict(r) for r in cur.fetchall()]
            return {"fil": filnavn, "sider": rows}
    finally:
        conn.close()


@app.post("/claims/{claim_id}/review")
def review_claim(claim_id: str, status: str = Form(...), comment: str = Form("")):
    if status not in ("confirmed", "disputed", "rejected", "unreviewed"):
        return {"error": "ugyldig status"}
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE claims SET review_status = %s, review_comment = %s WHERE claim_id = %s",
                (status, comment, claim_id)
            )
            conn.commit()
        return {"status": "ok", "claim_id": claim_id, "review_status": status}
    finally:
        conn.close()


def format_kilde(row: dict) -> str:
    kildetype = row.get("kildetype", "korpus").upper()
    filename = row.get("filename", "ukjent fil")
    s0 = row.get("side_start", "?")
    s1 = row.get("side_slutt", "?")
    sider = f"s. {s0}" if s0 == s1 else f"s. {s0}–{s1}"
    subdok = row.get("sub_dokumentid")
    deler = [f"[{kildetype}]", filename, sider]
    if subdok and subdok != "ukjent":
        deler.append(f"dok.id {subdok}")
    if row.get("doc_type"):
        deler.append(row["doc_type"])
    if row.get("person"):
        deler.append(row["person"])
    return " | ".join(deler)


@app.post("/analyze")
async def analyze(
    query: str = Form(...),
    analyse_type: str = Form("general"),
    filter_person: str = Form(""),
    filter_doctype: str = Form("")
):
    query_vector = embed(query)
    vec_str = vector_literal(query_vector)

    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
                SELECT text, filename, side_start, side_slutt, sub_dokumentid,
                       doc_type, person, kildetype,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE granularitet = 'fin'
            """
            params = [vec_str]
            if filter_person:
                sql += " AND person ILIKE %s"
                params.append(f"%{filter_person}%")
            if filter_doctype:
                sql += " AND doc_type = %s"
                params.append(filter_doctype)
            sql += " ORDER BY embedding <=> %s::vector LIMIT 15"
            params.append(vec_str)

            cur.execute(sql, params)
            results = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    context = "\n\n---\n\n".join([
        f"[{format_kilde(r)}]\n{r['text']}" for r in results
    ])

    prompts = {
        "motsetninger": f"""Analyser følgende kildemateriale og identifiser motstridende forklaringer eller påstander om: {query}

Vær konkret: (1) Hva påstår kilde A? (2) Hva påstår kilde B? (3) Hva er den faktiske motsetningen?
Oppgi alltid filnavn, dokumentid og sidetall for hver påstand.

KILDEMATERIALE:
{context}""",
        "tidslinje": f"""Sett opp en kronologisk analyse av alle hendelser relatert til: {query}

Marker eksplisitt hvor forklaringer eller bevis ikke stemmer med tidslinjen.

KILDEMATERIALE:
{context}""",
        "selvmotsigelser": f"""Analyser om samme person/kilde motsier seg selv angående: {query}

KILDEMATERIALE:
{context}""",
        "general": f"""Analyser følgende kildemateriale og svar på: {query}

Oppgi alltid filnavn, dokumentid og sidetall for påstander du baserer deg på.

KILDEMATERIALE:
{context}"""
    }

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompts.get(analyse_type, prompts["general"])}],
        max_tokens=2000
    )

    return {
        "analyse": response.choices[0].message.content,
        "kilder_brukt": len(results),
        "query": query
    }


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    analyse_type: Optional[str] = "general"
    filter_person: Optional[str] = ""
    filter_doctype: Optional[str] = ""

@app.post("/chat")
async def chat(req: ChatRequest):
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        return {"error": "Ingen brukermelding funnet"}

    latest_query = user_messages[-1].content
    query_vector = embed(latest_query)
    vec_str = vector_literal(query_vector)

    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
                SELECT text, filename, side_start, side_slutt, sub_dokumentid,
                       doc_type, person, kildetype,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE granularitet = 'fin'
            """
            params = [vec_str]
            if req.filter_person:
                sql += " AND person ILIKE %s"
                params.append(f"%{req.filter_person}%")
            if req.filter_doctype:
                sql += " AND doc_type = %s"
                params.append(req.filter_doctype)
            sql += " ORDER BY embedding <=> %s::vector LIMIT 14"
            params.append(vec_str)

            cur.execute(sql, params)
            results = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    context = "\n\n---\n\n".join([
        f"[{format_kilde(r)}]\n{r['text']}" for r in results
    ])

    system_prompt = f"""Du er en juridisk analytiker som jobber med norske rettsdokumenter.
Svar presist og kildebasert. For hver påstand du baserer deg på, oppgi:
- [KORPUS] eller [EGET]
- filnavn
- dokumentid hvis tilgjengelig
- sidetall
Analysetypen er: {req.analyse_type}

RELEVANT KILDEMATERIALE:
{context}

Hvis et spørsmål ikke kan besvares ut fra kildematerialet, si det eksplisitt."""

    openai_messages = [{"role": "system", "content": system_prompt}]
    for m in req.messages:
        openai_messages.append({"role": m.role, "content": m.content})

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=openai_messages,
        max_tokens=2000
    )

    return {
        "svar": response.choices[0].message.content,
        "kilder_brukt": len(results),
        "query": latest_query
    }
