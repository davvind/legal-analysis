from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import os, tempfile, json, re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import uuid
from pydantic import BaseModel
from typing import List, Optional

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

COLLECTION = "legal_docs"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
MAX_CHUNK_WORDS = 600
MIN_CHUNK_WORDS = 40


def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Returnerer liste av {page_num: int, text: str} per side.
    Bruker PyMuPDF direkte for sidesporring.
    """
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append({"page_num": page.number + 1, "text": text})
    doc.close()
    return pages


def split_into_paragraphs(text: str) -> list[str]:
    """
    Splitter tekst på tomme linjer eller linjeskift etterfulgt av stor bokstav.
    Renser støy-linjer (sidehoder, sidetall o.l.).
    """
    # Splitt på dobbelt linjeskift eller enkelt linjeskift etterfulgt av stor bokstav/tall
    raw = re.split(r'\n{2,}|\n(?=[A-ZÆØÅ0-9])', text)
    paragraphs = []
    for p in raw:
        p = p.strip()
        # Hopp over svært korte linjer (sidehoder, sidetall)
        if len(p.split()) < 5:
            continue
        paragraphs.append(p)
    return paragraphs


def build_chunks(pages: list[dict]) -> list[dict]:
    """
    Bygger chunks med sidesporring og avsnittbasert overlapp.

    Strategi:
    - Iterer over avsnitt med kunnskap om hvilken side de tilhører
    - Bygg chunk inntil MAX_CHUNK_WORDS er nådd
    - Overlapp: siste avsnitt fra forrige chunk inkluderes i neste
    - Hvert chunk lagrer side_start og side_slutt
    """
    # Flat liste av (avsnitt, side_num)
    all_paragraphs = []
    for page in pages:
        paras = split_into_paragraphs(page["text"])
        for para in paras:
            all_paragraphs.append({"text": para, "page": page["page_num"]})

    chunks = []
    i = 0
    last_para = None  # overlapp: siste avsnitt fra forrige chunk

    while i < len(all_paragraphs):
        chunk_paras = []
        chunk_pages = set()
        word_count = 0

        # Inkluder overlapp fra forrige chunk
        if last_para is not None:
            chunk_paras.append(last_para["text"])
            chunk_pages.add(last_para["page"])
            word_count += len(last_para["text"].split())

        # Fyll chunk inntil MAX_CHUNK_WORDS
        while i < len(all_paragraphs):
            para = all_paragraphs[i]
            para_words = len(para["text"].split())

            # Hvis dette avsnittet alene er for langt, splitt det på setninger
            if para_words > MAX_CHUNK_WORDS:
                sentences = re.split(r'(?<=[.!?])\s+', para["text"])
                for sent in sentences:
                    sent_words = len(sent.split())
                    if word_count + sent_words > MAX_CHUNK_WORDS and word_count >= MIN_CHUNK_WORDS:
                        break
                    chunk_paras.append(sent)
                    chunk_pages.add(para["page"])
                    word_count += sent_words
                i += 1
                break

            if word_count + para_words > MAX_CHUNK_WORDS and word_count >= MIN_CHUNK_WORDS:
                break

            chunk_paras.append(para["text"])
            chunk_pages.add(para["page"])
            word_count += para_words
            i += 1

        if not chunk_paras:
            i += 1
            continue

        chunk_text = "\n\n".join(chunk_paras)
        sorted_pages = sorted(chunk_pages)

        chunks.append({
            "text": chunk_text,
            "side_start": sorted_pages[0],
            "side_slutt": sorted_pages[-1],
        })

        # Sett overlapp: siste avsnitt i denne chunken
        last_para = all_paragraphs[i - 1] if i > 0 else None

    return chunks


def embed(text):
    response = openai_client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding


def format_kilde(payload: dict) -> str:
    """Formater kildevisning: filnavn + sideintervall."""
    filename = payload.get("filename", "ukjent fil")
    side_start = payload.get("side_start", "?")
    side_slutt = payload.get("side_slutt", "?")
    doc_type = payload.get("doc_type", "")
    person = payload.get("person", "")

    if side_start == side_slutt:
        sider = f"s. {side_start}"
    else:
        sider = f"s. {side_start}–{side_slutt}"

    deler = [filename, sider]
    if doc_type:
        deler.append(doc_type)
    if person:
        deler.append(person)

    return " | ".join(deler)


@app.get("/")
def root():
    return {"status": "ok", "message": "Legal Analysis API kjører"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    doc_type: str = Form("ukjent"),
    person: str = Form(""),
    dato: str = Form("")
):
    ensure_collection()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pages = extract_pages(tmp_path)
    finally:
        os.unlink(tmp_path)

    chunks = build_chunks(pages)

    points = []
    for chunk in chunks:
        if len(chunk["text"].strip()) < 50:
            continue
        vector = embed(chunk["text"])
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk["text"],
                "filename": file.filename,
                "doc_type": doc_type,
                "person": person,
                "dato": dato,
                "side_start": chunk["side_start"],
                "side_slutt": chunk["side_slutt"],
            }
        ))

    qdrant.upsert(collection_name=COLLECTION, points=points)
    return {
        "status": "ok",
        "chunks_indexed": len(points),
        "filename": file.filename,
        "sider_prosessert": len(pages)
    }


@app.post("/analyze")
async def analyze(
    query: str = Form(...),
    analyse_type: str = Form("general"),
    filter_person: str = Form(""),
    filter_doctype: str = Form("")
):
    query_vector = embed(query)

    search_filter = None
    if filter_person or filter_doctype:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        if filter_person:
            conditions.append(FieldCondition(key="person", match=MatchValue(value=filter_person)))
        if filter_doctype:
            conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=filter_doctype)))
        search_filter = Filter(must=conditions)

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=15,
        query_filter=search_filter
    )

    context = "\n\n---\n\n".join([
        f"[{format_kilde(r.payload)}]\n{r.payload['text']}"
        for r in results
    ])

    prompts = {
        "motsetninger": f"""Analyser følgende kildemateriale og identifiser motstridende forklaringer eller påstander om: {query}

Vær konkret: (1) Hva påstår kilde A? (2) Hva påstår kilde B? (3) Hva er den faktiske motsetningen?
Oppgi alltid filnavn og sidetall for hver påstand.

KILDEMATERIALE:
{context}""",
        "tidslinje": f"""Sett opp en kronologisk analyse av alle hendelser relatert til: {query}

Marker eksplisitt hvor forklaringer eller bevis ikke stemmer med tidslinjen.
Oppgi alltid filnavn og sidetall for hver hendelse.

KILDEMATERIALE:
{context}""",
        "selvmotsigelser": f"""Analyser om samme person/kilde motsier seg selv angående: {query}

Vis konkret hva som ble sagt, i hvilket dokument og på hvilken side, og hva som er selvmotsigende.

KILDEMATERIALE:
{context}""",
        "general": f"""Analyser følgende kildemateriale og svar på: {query}

Oppgi alltid filnavn og sidetall for påstander du baserer deg på.

KILDEMATERIALE:
{context}"""
    }

    prompt = prompts.get(analyse_type, prompts["general"])

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )

    return {
        "analyse": response.choices[0].message.content,
        "kilder_brukt": len(results),
        "query": query
    }


# --- Chat endpoint ---

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

    search_filter = None
    if req.filter_person or req.filter_doctype:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        if req.filter_person:
            conditions.append(FieldCondition(key="person", match=MatchValue(value=req.filter_person)))
        if req.filter_doctype:
            conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=req.filter_doctype)))
        search_filter = Filter(must=conditions)

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=12,
        query_filter=search_filter
    )

    context = "\n\n---\n\n".join([
        f"[{format_kilde(r.payload)}]\n{r.payload['text']}"
        for r in results
    ])

    system_prompt = f"""Du er en juridisk analytiker som jobber med norske rettsdokumenter.
Svar presist og kildebasert. Oppgi alltid filnavn og sidetall (f.eks. "dok6_del2.pdf, s. 45") for hver påstand du baserer deg på.
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

    assistant_reply = response.choices[0].message.content

    return {
        "svar": assistant_reply,
        "kilder_brukt": len(results),
        "query": latest_query
    }
