from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os, tempfile, re, hashlib
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from openai import OpenAI
import uuid
import psycopg2
import psycopg2.extras
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

load_dotenv()

app = FastAPI(title="Legal Analysis API")

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
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

COLLECTION_KORPUS = "legal_docs"
COLLECTION_EGET   = "egenproduksjon"
EMBED_MODEL       = "text-embedding-3-large"
EMBED_DIM         = 3072
MAX_WORDS_FIN     = 400
MAX_WORDS_GROV    = 1500
MAX_WORDS_EGET    = 250
MIN_WORDS         = 40
DEFAULT_CASE_NAME = "orderud"


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def db():
    """Ny tilkobling per request — enkelt og robust for FastAPI."""
    return psycopg2.connect(os.environ["DATABASE_URL"])


def ensure_default_case() -> str:
    """Henter eller oppretter standard-saken. Returnerer case_id som str."""
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


def start_processing_run(run_type: str, model_name: str, model_version: str = "",
                          prompt_version: str = "") -> str:
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO processing_runs (run_type, model_name, model_version, prompt_version, status)
                   VALUES (%s, %s, %s, %s, 'running') RETURNING run_id""",
                (run_type, model_name, model_version, prompt_version)
            )
            run_id = cur.fetchone()[0]
            conn.commit()
            return str(run_id)
    finally:
        conn.close()


def complete_processing_run(run_id: str, status: str = "completed", notes: str = ""):
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE processing_runs SET completed_at = %s, status = %s, notes = %s
                   WHERE run_id = %s""",
                (datetime.utcnow(), status, notes, run_id)
            )
            conn.commit()
    finally:
        conn.close()


def insert_source_file(case_id: str, filename: str, checksum: str, pdf_id: str,
                        del_nummer: int, file_size: int, doc_type: str,
                        person: str, dato: str) -> str:
    """Inserter source_file. Hvis checksum allerede finnes, returneres eksisterende file_id."""
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
                    file_size_bytes, doc_type, person, dato)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING file_id""",
                (case_id, filename, checksum, pdf_id, del_nummer, file_size, doc_type, person, dato)
            )
            file_id = cur.fetchone()[0]
            conn.commit()
            return str(file_id)
    finally:
        conn.close()


def insert_page(file_id: str, page_number: int, ocr_text: str,
                 sub_dokumentid: Optional[str], azure_run_id: str) -> str:
    conn = db()
    try:
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
    finally:
        conn.close()


def insert_claim(page_id: str, run_id: str, speaker: str, speaker_role: str,
                  claim_text: str, normalized_claim: str, claim_type: str,
                  topic: str, event_date: Optional[str], confidence: float) -> str:
    conn = db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO claims
                   (page_id, run_id, speaker, speaker_role, claim_text, normalized_claim,
                    claim_type, topic, event_date, extraction_confidence)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING claim_id""",
                (page_id, run_id, speaker, speaker_role, claim_text, normalized_claim,
                 claim_type, topic, event_date, confidence)
            )
            claim_id = cur.fetchone()[0]
            conn.commit()
            return str(claim_id)
    finally:
        conn.close()


def get_page_context(page_id: str) -> dict:
    """Henter metadata om en side for kildevisning."""
    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT p.page_number, p.sub_dokumentid, sf.original_filename,
                       sf.doc_type, sf.person
                FROM pages p
                JOIN source_files sf ON sf.file_id = p.file_id
                WHERE p.page_id = %s
            """, (page_id,))
            row = cur.fetchone()
            return dict(row) if row else {}
    finally:
        conn.close()


def get_claim_context(claim_id: str) -> dict:
    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT c.speaker, c.speaker_role, c.claim_text, c.normalized_claim,
                       c.claim_type, c.topic, c.event_date, c.review_status,
                       p.page_number, sf.original_filename
                FROM claims c
                JOIN pages p ON p.page_id = c.page_id
                JOIN source_files sf ON sf.file_id = p.file_id
                WHERE c.claim_id = %s
            """, (claim_id,))
            row = cur.fetchone()
            return dict(row) if row else {}
    finally:
        conn.close()


# ── Dokumentid-deteksjon (korpus) ─────────────────────────────────────────────

def decode_doubled(text: str) -> str:
    result = []
    i = 0
    while i < len(text) - 1:
        if text[i] == text[i + 1] and text[i].isalnum():
            result.append(text[i])
            i += 2
        else:
            result.append(text[i])
            i += 1
    if i < len(text):
        result.append(text[i])
    return ''.join(result)


def extract_footer_dokumentid(footer_text: str) -> Optional[str]:
    decoded = decode_doubled(footer_text)
    match = re.search(r'dokumentid[:\s]*(\d+)', decoded, re.IGNORECASE)
    return match.group(1) if match else None


# ── Azure Document Intelligence ───────────────────────────────────────────────

def extract_with_azure(pdf_path: str) -> list[dict]:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(
        os.environ["AZURE_DI_ENDPOINT"],
        AzureKeyCredential(os.environ["AZURE_DI_KEY"])
    )

    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-read",
            analyze_request=f,
            content_type="application/octet-stream"
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


# ── Segmentering i sub-dokumenter (korpus) ────────────────────────────────────

def segment_into_subdocs(pages: list[dict]) -> list[dict]:
    subdocs = []
    current_id = None
    current_pages = []

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
        chunk_paras = []
        chunk_pages = set()
        word_count = 0

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


def build_chunks_eget(pages: list[dict]) -> list[dict]:
    return build_chunks(pages, MAX_WORDS_EGET, "fin")


# ── Cross-file stitching ──────────────────────────────────────────────────────

def hent_forrige_del_kontekst(pdf_id: str, del_nummer: int) -> str:
    if del_nummer <= 1:
        return ""
    try:
        results, _ = qdrant.scroll(
            collection_name=COLLECTION_KORPUS,
            scroll_filter=Filter(must=[
                FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id)),
                FieldCondition(key="del_nummer", match=MatchValue(value=del_nummer - 1)),
                FieldCondition(key="granularitet", match=MatchValue(value="fin")),
            ]),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return ""
        siste = max(results, key=lambda r: r.payload.get("side_slutt", 0))
        tekst = siste.payload.get("text", "")
        avsnitt = [p for p in tekst.split("\n\n") if p.strip()]
        return avsnitt[-1] if avsnitt else ""
    except Exception:
        return ""


# ── Claim-ekstraksjon ──────────────────────────────────────────────────────────

CLAIM_EXTRACTION_PROMPT = """Du analyserer en side fra et norsk rettsdokument. Identifiser konkrete \
påstander (claims) i teksten — utsagn der noen hevder noe om en hendelse, observasjon eller fakta.

For hver påstand du finner, returner et JSON-objekt med feltene:
- speaker: hvem som hevder dette (navn hvis kjent, ellers rolle som "vitne", "tiltalt", "politi")
- speaker_role: en av "vitne", "tiltalt", "politi", "sakkyndig", "domstol", "ukjent"
- claim_text: direkte utdrag fra teksten som påstanden bygger på
- normalized_claim: kort, presis omformulering av påstanden i tredjeperson
- claim_type: en av "observasjon", "forklaring", "påstand", "innrømmelse", "benektelse"
- topic: kort tema-tag, f.eks. "bilobservasjon", "tidspunkt", "alibi"
- event_date: dato hendelsen fant sted i format YYYY-MM-DD, eller null hvis ukjent
- extraction_confidence: din egen vurdering 0.0-1.0 av hvor sikker ekstraksjonen er

Returner KUN en JSON-liste med disse objektene, ingen annen tekst. Hvis ingen klare påstander \
finnes på siden, returner tom liste [].

SIDETEKST:
{text}"""


def extract_claims_from_page(page_text: str) -> list[dict]:
    """Kjører claim-ekstraksjon på én side via GPT-4o-mini. Returnerer liste av claim-dicts."""
    import json

    if len(page_text.split()) < 15:
        return []

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": CLAIM_EXTRACTION_PROMPT.format(text=page_text)}],
            max_tokens=1500,
            response_format={"type": "json_object"} if False else None,
        )
        raw = response.choices[0].message.content.strip()
        # Modellen kan pakke svaret i ```json ... ``` til tross for instruks
        raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        # Forvent enten ren liste eller {"claims": [...]}
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("claims", [])
        if not isinstance(parsed, list):
            return []
        return parsed
    except Exception:
        return []


# ── Embedding og kildeformat ──────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


def format_kilde(payload: dict, kilde_type: str = "KORPUS") -> str:
    filename = payload.get("filename", "ukjent fil")
    s0 = payload.get("side_start", "?")
    s1 = payload.get("side_slutt", "?")
    sider = f"s. {s0}" if s0 == s1 else f"s. {s0}–{s1}"
    subdok = payload.get("sub_dokumentid")
    deler = [f"[{kilde_type}]", filename, sider]
    if subdok and subdok != "ukjent":
        deler.append(f"dok.id {subdok}")
    if payload.get("doc_type"):
        deler.append(payload["doc_type"])
    if payload.get("person"):
        deler.append(payload["person"])
    return " | ".join(deler)


def ensure_collection(name: str):
    existing = [c.name for c in qdrant.get_collections().collections]
    if name not in existing:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )


# ── Endepunkter ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Legal Analysis API kjører"}


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
    extract_claims: bool = Form(True),
):
    """
    Laster opp korpusdokument.
    Skriver råstoff (source_files, pages) til PostgreSQL.
    Kjører claim-ekstraksjon som egen processing_run.
    Indekserer i Qdrant med pekere til PostgreSQL i payload.
    """
    ensure_collection(COLLECTION_KORPUS)
    case_id = ensure_default_case()

    content = await file.read()
    checksum = hashlib.sha256(content).hexdigest()
    file_size = len(content)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    azure_run_id = start_processing_run("ocr", "azure-di-prebuilt-read")

    try:
        pages = extract_with_azure(tmp_path)
        complete_processing_run(azure_run_id, "completed", f"{len(pages)} sider")
    except Exception as e:
        complete_processing_run(azure_run_id, "failed", str(e))
        os.unlink(tmp_path)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Skriv source_file til PostgreSQL (idempotent via checksum)
    file_id = insert_source_file(
        case_id, file.filename, checksum, pdf_id or file.filename,
        del_nummer, file_size, doc_type, person, dato
    )

    # Skriv hver side til PostgreSQL (råstoff)
    pg_page_ids = {}  # page_num -> page_id
    for page in pages:
        pg_page_id = insert_page(file_id, page["page_num"], page["text"],
                                  page["dokumentid"], azure_run_id)
        pg_page_ids[page["page_num"]] = pg_page_id

    # Claim-ekstraksjon som egen processing_run
    claim_run_id = None
    claims_extracted = 0
    if extract_claims:
        claim_run_id = start_processing_run("claim_extraction", "gpt-4o-mini",
                                             prompt_version="v1")
        for page in pages:
            page_claims = extract_claims_from_page(page["text"])
            pg_page_id = pg_page_ids.get(page["page_num"])
            for c in page_claims:
                try:
                    insert_claim(
                        pg_page_id, claim_run_id,
                        c.get("speaker", ""), c.get("speaker_role", "ukjent"),
                        c.get("claim_text", ""), c.get("normalized_claim", ""),
                        c.get("claim_type", ""), c.get("topic", ""),
                        c.get("event_date"), c.get("extraction_confidence", 0.5)
                    )
                    claims_extracted += 1
                except Exception:
                    continue
        complete_processing_run(claim_run_id, "completed", f"{claims_extracted} claims")

    # Cross-file stitching og chunking (uendret logikk)
    prefix_context = hent_forrige_del_kontekst(pdf_id, del_nummer) if pdf_id and del_nummer > 1 else ""
    subdocs = segment_into_subdocs(pages)

    points = []
    chunks_fin_total = 0
    chunks_grov_total = 0

    for idx, subdoc in enumerate(subdocs):
        ctx = prefix_context if idx == 0 else ""
        fine_chunks = build_chunks(subdoc["sider"], MAX_WORDS_FIN, "fin", ctx)
        grov_chunks = build_chunks(subdoc["sider"], MAX_WORDS_GROV, "grov", ctx)
        chunks_fin_total += len(fine_chunks)
        chunks_grov_total += len(grov_chunks)

        for chunk in fine_chunks + grov_chunks:
            if len(chunk["text"].strip()) < 50:
                continue
            # Finn nærmeste page_id for denne chunken (side_start)
            pg_page_id = pg_page_ids.get(chunk["side_start"])
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(chunk["text"]),
                payload={
                    "text": chunk["text"],
                    "filename": file.filename,
                    "pdf_id": pdf_id or file.filename,
                    "del_nummer": del_nummer,
                    "sub_dokumentid": subdoc["dokumentid"],
                    "doc_type": doc_type,
                    "person": person,
                    "dato": dato,
                    "side_start": chunk["side_start"],
                    "side_slutt": chunk["side_slutt"],
                    "granularitet": chunk["granularitet"],
                    # Pekere til PostgreSQL (autoritativ kilde)
                    "case_id": case_id,
                    "file_id": file_id,
                    "page_id": pg_page_id,
                }
            ))

    qdrant.upsert(collection_name=COLLECTION_KORPUS, points=points)

    return {
        "status": "ok",
        "chunks_fin": chunks_fin_total,
        "chunks_grov": chunks_grov_total,
        "sub_dokumenter": len(subdocs),
        "sider_prosessert": len(pages),
        "filename": file.filename,
        "cross_file_stitching": bool(prefix_context),
        "claims_extracted": claims_extracted,
        "file_id": file_id,
        "case_id": case_id,
    }


@app.post("/upload/egen")
async def upload_egen(
    file: UploadFile = File(...),
    doc_type: str = Form("egenproduksjon"),
    person: str = Form(""),
    dato: str = Form(""),
    tittel: str = Form(""),
):
    ensure_collection(COLLECTION_EGET)
    case_id = ensure_default_case()

    content = await file.read()
    checksum = hashlib.sha256(content).hexdigest()
    file_size = len(content)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    azure_run_id = start_processing_run("ocr", "azure-di-prebuilt-read")
    try:
        pages = extract_with_azure(tmp_path)
        complete_processing_run(azure_run_id, "completed", f"{len(pages)} sider")
    except Exception as e:
        complete_processing_run(azure_run_id, "failed", str(e))
        os.unlink(tmp_path)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    file_id = insert_source_file(
        case_id, file.filename, checksum, tittel or file.filename,
        1, file_size, doc_type, person, dato
    )

    pg_page_ids = {}
    for page in pages:
        pg_page_id = insert_page(file_id, page["page_num"], page["text"], None, azure_run_id)
        pg_page_ids[page["page_num"]] = pg_page_id

    chunks = build_chunks_eget(pages)
    points = []

    for chunk in chunks:
        if len(chunk["text"].strip()) < 30:
            continue
        pg_page_id = pg_page_ids.get(chunk["side_start"])
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embed(chunk["text"]),
            payload={
                "text": chunk["text"],
                "filename": file.filename,
                "tittel": tittel or file.filename,
                "doc_type": doc_type,
                "person": person,
                "dato": dato,
                "side_start": chunk["side_start"],
                "side_slutt": chunk["side_slutt"],
                "granularitet": "fin",
                "case_id": case_id,
                "file_id": file_id,
                "page_id": pg_page_id,
            }
        ))

    qdrant.upsert(collection_name=COLLECTION_EGET, points=points)

    return {
        "status": "ok",
        "chunks_indeksert": len(points),
        "sider_prosessert": len(pages),
        "filename": file.filename,
        "file_id": file_id,
    }


@app.get("/claims")
def list_claims(speaker: str = "", review_status: str = "", limit: int = 50):
    """Lister claims med valgfri filtrering — for fremtidig review-UI."""
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
    """
    Viser claims ekstrahert fra en spesifikk side sammen med rå OCR-tekst
    fra samme side, for manuell kvalitetskontroll.
    Bruk: /claims/sammenlign?filnavn=dok6_del1.pdf&side=7
    """
    conn = db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Hent siden
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

            # Hent alle claims fra denne siden
            cur.execute("""
                SELECT claim_id, speaker, speaker_role, claim_text,
                       normalized_claim, claim_type, topic, event_date,
                       extraction_confidence, review_status, review_comment
                FROM claims
                WHERE page_id = %s
                ORDER BY created_at
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
    """
    Gir en kompakt oversikt over alle sider i en fil og antall claims per side.
    Nyttig for å raskt finne sider verdt å sjekke manuelt.
    Bruk: /claims/fil-oversikt?filnavn=dok6_del1.pdf
    """
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
    """Setter review_status på en claim: confirmed/disputed/rejected."""
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


@app.post("/analyze")
async def analyze(
    query: str = Form(...),
    analyse_type: str = Form("general"),
    filter_person: str = Form(""),
    filter_doctype: str = Form("")
):
    query_vector = embed(query)

    conditions_korpus = [FieldCondition(key="granularitet", match=MatchValue(value="fin"))]
    if filter_person:
        conditions_korpus.append(FieldCondition(key="person", match=MatchValue(value=filter_person)))
    if filter_doctype:
        conditions_korpus.append(FieldCondition(key="doc_type", match=MatchValue(value=filter_doctype)))

    korpus_results = qdrant.search(
        collection_name=COLLECTION_KORPUS,
        query_vector=query_vector,
        limit=12,
        query_filter=Filter(must=conditions_korpus)
    )

    try:
        eget_results = qdrant.search(collection_name=COLLECTION_EGET, query_vector=query_vector, limit=5)
    except Exception:
        eget_results = []

    context_korpus = "\n\n---\n\n".join([
        f"[{format_kilde(r.payload, 'KORPUS')}]\n{r.payload['text']}"
        for r in korpus_results
    ])
    context_eget = "\n\n---\n\n".join([
        f"[{format_kilde(r.payload, 'EGET')}]\n{r.payload['text']}"
        for r in eget_results
    ])
    context = context_korpus
    if context_eget:
        context += f"\n\n{'='*40}\nEGENPRODUSERT MATERIALE:\n{'='*40}\n\n{context_eget}"

    prompts = {
        "motsetninger": f"""Analyser følgende kildemateriale og identifiser motstridende forklaringer eller påstander om: {query}

Vær konkret: (1) Hva påstår kilde A? (2) Hva påstår kilde B? (3) Hva er den faktiske motsetningen?
Oppgi alltid [KORPUS] eller [EGET], filnavn, dokumentid og sidetall for hver påstand.

KILDEMATERIALE:
{context}""",
        "tidslinje": f"""Sett opp en kronologisk analyse av alle hendelser relatert til: {query}

Marker eksplisitt hvor forklaringer eller bevis ikke stemmer med tidslinjen.
Oppgi alltid [KORPUS] eller [EGET], filnavn, dokumentid og sidetall.

KILDEMATERIALE:
{context}""",
        "selvmotsigelser": f"""Analyser om samme person/kilde motsier seg selv angående: {query}

Vis konkret hva som ble sagt, i hvilket dokument og på hvilken side, og hva som er selvmotsigende.
Merk tydelig om kilden er [KORPUS] eller [EGET].

KILDEMATERIALE:
{context}""",
        "general": f"""Analyser følgende kildemateriale og svar på: {query}

Oppgi alltid [KORPUS] eller [EGET], filnavn, dokumentid og sidetall for påstander du baserer deg på.

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
        "kilder_brukt": len(korpus_results) + len(eget_results),
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

    conditions_korpus = [FieldCondition(key="granularitet", match=MatchValue(value="fin"))]
    if req.filter_person:
        conditions_korpus.append(FieldCondition(key="person", match=MatchValue(value=req.filter_person)))
    if req.filter_doctype:
        conditions_korpus.append(FieldCondition(key="doc_type", match=MatchValue(value=req.filter_doctype)))

    korpus_results = qdrant.search(
        collection_name=COLLECTION_KORPUS,
        query_vector=query_vector,
        limit=10,
        query_filter=Filter(must=conditions_korpus)
    )

    try:
        eget_results = qdrant.search(collection_name=COLLECTION_EGET, query_vector=query_vector, limit=4)
    except Exception:
        eget_results = []

    context_korpus = "\n\n---\n\n".join([
        f"[{format_kilde(r.payload, 'KORPUS')}]\n{r.payload['text']}"
        for r in korpus_results
    ])
    context_eget = "\n\n---\n\n".join([
        f"[{format_kilde(r.payload, 'EGET')}]\n{r.payload['text']}"
        for r in eget_results
    ])
    context = context_korpus
    if context_eget:
        context += f"\n\n{'='*40}\nEGENPRODUSERT MATERIALE:\n{'='*40}\n\n{context_eget}"

    system_prompt = f"""Du er en juridisk analytiker som jobber med norske rettsdokumenter.
Svar presist og kildebasert. For hver påstand du baserer deg på, oppgi:
- [KORPUS] eller [EGET]
- filnavn
- dokumentid hvis tilgjengelig (f.eks. "dok.id 83654328")
- sidetall (f.eks. "s. 23")
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
        "kilder_brukt": len(korpus_results) + len(eget_results),
        "query": latest_query
    }
