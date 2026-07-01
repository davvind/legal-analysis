from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os, hashlib, json, tempfile
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

# CORS: ikke lenger "*". Sett ALLOWED_ORIGINS (komma-separert) i miljøet.
# Prod-eksempel: ALLOWED_ORIGINS=https://davvind.github.io
_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
)
ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
    max_age=3600,
)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DEFAULT_CASE_NAME = "orderud"
BLOB_CONTAINER    = "dokumenter"
SERVICEBUS_QUEUE  = "dokument-prosessering"
EMBED_MODEL       = "text-embedding-3-large"   # 3072 dim, indekseres via halfvec-cast
EMBED_DIM         = 3072
UPLOAD_CHUNK      = 1024 * 1024  # 1 MB — streaming-blokk for opplasting


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def db():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def ensure_default_case() -> str:
    """Idempotent via uq_cases_case_name (migrasjon_002)."""
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cases (case_name, description) VALUES (%s, %s) "
                "ON CONFLICT (case_name) DO NOTHING",
                (DEFAULT_CASE_NAME, "Standardsak, opprettet automatisk")
            )
            conn.commit()
            cur.execute("SELECT case_id FROM cases WHERE case_name = %s", (DEFAULT_CASE_NAME,))
            return str(cur.fetchone()[0])
    finally:
        conn.close()


def find_existing_source_file(checksum: str) -> Optional[dict]:
    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT file_id, blob_path FROM source_files WHERE sha256_checksum = %s",
                (checksum,)
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def has_completed_job(file_id: str) -> bool:
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM processing_jobs "
                "WHERE file_id = %s AND status = 'completed' LIMIT 1",
                (file_id,)
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def insert_source_file_pending(case_id: str, filename: str, checksum: str, pdf_id: str,
                                del_nummer: int, file_size: int, doc_type: str,
                                person: str, dato: str, blob_path: str) -> str:
    conn = db()
    try:
        with conn.cursor() as cur:
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


# ── Azure Blob Storage (streaming) ────────────────────────────────────────────

def upload_stream_to_blob(local_path: str, blob_path: str) -> None:
    """Laster opp fra fil på disk som stream — holder ikke hele filen i RAM."""
    blob_service = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    )
    blob_client = blob_service.get_blob_client(container=BLOB_CONTAINER, blob=blob_path)
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True, max_concurrency=4)


# ── Azure Service Bus ─────────────────────────────────────────────────────────

def send_job_message(job_id: str, file_id: str, case_id: str, blob_path: str,
                      job_type: str = "full_ingest") -> None:
    sb_client = ServiceBusClient.from_connection_string(
        os.environ["AZURE_SERVICEBUS_CONNECTION_STRING"]
    )
    payload = {
        "job_id": job_id, "file_id": file_id, "case_id": case_id,
        "blob_path": blob_path, "job_type": job_type,
    }
    with sb_client:
        sender = sb_client.get_queue_sender(queue_name=SERVICEBUS_QUEUE)
        with sender:
            sender.send_messages(ServiceBusMessage(json.dumps(payload)))


# ── Embedding (kun for søk) ───────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    # 3072 dim — samme modell som worker. Ingen dimensions-parameter (halfvec-rute).
    response = openai_client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


def vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


# ── Retrieval (delt av /analyze og /chat) ─────────────────────────────────────

def retrieve_chunks(vec_str: str, case_id: str, filter_person: str = "",
                    filter_doctype: str = "", limit: int = 15) -> list[dict]:
    """
    Schema-korrekt henting:
      * JOIN source_files for original_filename (chunks har ikke filename).
      * halfvec-cast identisk med HNSW-indeksen (ellers brukes ikke indeksen).
      * is_active = TRUE (ser aldri superseded chunks fra reingest).
      * case_id-scoping (hindrer lekkasje mellom saker når multi-case kommer).
    """
    sql = """
        SELECT c.text,
               sf.original_filename AS filename,
               c.side_start, c.side_slutt, c.sub_dokumentid,
               c.doc_type, c.person, c.kildetype,
               1 - (c.embedding::halfvec(3072) <=> %s::halfvec(3072)) AS similarity
        FROM chunks c
        JOIN source_files sf ON sf.file_id = c.file_id
        WHERE c.granularitet = 'fin'
          AND c.is_active = TRUE
          AND c.case_id = %s
    """
    params: list = [vec_str, case_id]
    if filter_person:
        sql += " AND c.person ILIKE %s"; params.append(f"%{filter_person}%")
    if filter_doctype:
        sql += " AND c.doc_type = %s"; params.append(filter_doctype)
    sql += " ORDER BY c.embedding::halfvec(3072) <=> %s::halfvec(3072) LIMIT %s"
    params += [vec_str, limit]

    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
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
    Streaming-opplasting. Filen spooles til disk mens SHA256 beregnes (aldri hele
    filen i RAM), dedupliseres på sjekksum, streames til Blob, og en jobb legges i kø.
    Returnerer umiddelbart.
    """
    case_id = ensure_default_case()

    # 1. Spool til temp + hash, blokk for blokk.
    hasher = hashlib.sha256()
    file_size = 0
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    try:
        while True:
            block = await file.read(UPLOAD_CHUNK)
            if not block:
                break
            hasher.update(block)
            file_size += len(block)
            tmp.write(block)
        tmp.close()
        checksum = hasher.hexdigest()

        # 2. Dedup: samme fil finnes fra før?
        existing = find_existing_source_file(checksum)
        if existing:
            file_id   = str(existing["file_id"])
            blob_path = existing["blob_path"]
            if has_completed_job(file_id):
                # Allerede ferdig indeksert — ikke last opp eller reprosesser.
                return {
                    "status": "already_ingested",
                    "file_id": file_id,
                    "case_id": case_id,
                    "filename": file.filename,
                    "blob_path": blob_path,
                    "message": "Filen er allerede indeksert. Ingen ny jobb opprettet.",
                }
            # Finnes, men ingen fullført jobb -> requeue mot eksisterende blob/file_id.
            # (Reingest er idempotent i worker v4, så re-opplasting av blob er unødvendig.)
        else:
            # 3. Ny fil: stream til Blob, skriv source_file-rad.
            blob_path = f"{DEFAULT_CASE_NAME}/{pdf_id or 'ukjent'}/del{del_nummer}_{file.filename}"
            upload_stream_to_blob(tmp_path, blob_path)
            file_id = insert_source_file_pending(
                case_id, file.filename, checksum, pdf_id or file.filename,
                del_nummer, file_size, doc_type, person, dato, blob_path
            )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # 4. Opprett jobb og legg i kø.
    job_id = insert_processing_job(case_id, file_id, "full_ingest", blob_path)
    send_job_message(job_id, file_id, case_id, blob_path)

    return {
        "status": "queued",
        "job_id": job_id,
        "file_id": file_id,
        "case_id": case_id,
        "filename": file.filename,
        "blob_path": blob_path,
        "message": "Filen er lastet opp og lagt i kø. Følg fremdrift via /upload/status/{job_id}.",
    }


@app.get("/upload/status/{job_id}")
def upload_status(job_id: str):
    status = get_job_status(job_id)
    if not status:
        return {"error": "Fant ingen jobb med denne IDen"}
    return status


@app.get("/upload/status")
def upload_status_for_case(case_id: str = "", limit: int = 20):
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
                query += " AND j.case_id = %s"; params.append(case_id)
            query += " ORDER BY j.queued_at DESC LIMIT %s"; params.append(limit)
            cur.execute(query, params)
            return {"jobs": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()


# ── Claims (uendret review-flyt) ──────────────────────────────────────────────

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
                query += " AND c.speaker ILIKE %s"; params.append(f"%{speaker}%")
            if review_status:
                query += " AND c.review_status = %s"; params.append(review_status)
            query += " ORDER BY c.created_at DESC LIMIT %s"; params.append(limit)
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


# ── Analyse ───────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    query: str = Form(...),
    analyse_type: str = Form("general"),
    filter_person: str = Form(""),
    filter_doctype: str = Form("")
):
    case_id = ensure_default_case()
    vec_str = vector_literal(embed(query))
    results = retrieve_chunks(vec_str, case_id, filter_person, filter_doctype, limit=15)

    context = "\n\n---\n\n".join([f"[{format_kilde(r)}]\n{r['text']}" for r in results])

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
        "query": query,
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

    case_id = ensure_default_case()
    latest_query = user_messages[-1].content
    vec_str = vector_literal(embed(latest_query))
    results = retrieve_chunks(vec_str, case_id, req.filter_person, req.filter_doctype, limit=14)

    context = "\n\n---\n\n".join([f"[{format_kilde(r)}]\n{r['text']}" for r in results])

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
        "query": latest_query,
    }
