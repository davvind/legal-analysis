from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os, tempfile
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
MAX_WORDS_FIN = 400    # Qdrant/chat-søk
MAX_WORDS_GROV = 1500  # GraphRAG
MIN_WORDS = 40


def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )


def extract_with_azure(pdf_path: str) -> list[dict]:
    """
    Bruker Azure Document Intelligence Read-modellen.
    Returnerer liste av {page_num, text} per side,
    med avsnitt rekonstruert fra linje-outputs.
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    endpoint = os.environ["AZURE_DI_ENDPOINT"]
    key = os.environ["AZURE_DI_KEY"]

    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-read",
            analyze_request=f,
            content_type="application/octet-stream"
        )
    result = poller.result()

    pages = []
    for page in result.pages:
        page_num = page.page_number
        lines = [line.content for line in (page.lines or [])]
        text = "\n".join(lines)
        if text.strip():
            pages.append({"page_num": page_num, "text": text})

    return pages


def build_chunks(pages: list[dict], max_words: int, granularitet: str) -> list[dict]:
    """
    Avsnittbasert chunking med sidesporring og overlapp.
    Splitter på tomme linjer; overlapp = siste avsnitt fra forrige chunk.
    """
    import re

    # Flat liste av (avsnitt, side)
    all_paras = []
    for page in pages:
        raw = re.split(r'\n{2,}|\n(?=[A-ZÆØÅ0-9])', page["text"])
        for p in raw:
            p = p.strip()
            if len(p.split()) < 5:
                continue
            all_paras.append({"text": p, "page": page["page_num"]})

    chunks = []
    i = 0
    last_para = None

    while i < len(all_paras):
        chunk_paras = []
        chunk_pages = set()
        word_count = 0

        # Overlapp fra forrige chunk
        if last_para is not None:
            chunk_paras.append(last_para["text"])
            chunk_pages.add(last_para["page"])
            word_count += len(last_para["text"].split())

        while i < len(all_paras):
            para = all_paras[i]
            pw = len(para["text"].split())

            # Langt avsnitt: splitt på setninger
            if pw > max_words:
                import re as re2
                sentences = re2.split(r'(?<=[.!?])\s+', para["text"])
                for sent in sentences:
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

        last_para = all_paras[i - 1] if i > 0 else None

    return chunks


def embed(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


def format_kilde(payload: dict) -> str:
    filename = payload.get("filename", "ukjent fil")
    s0 = payload.get("side_start", "?")
    s1 = payload.get("side_slutt", "?")
    sider = f"s. {s0}" if s0 == s1 else f"s. {s0}–{s1}"
    deler = [filename, sider]
    if payload.get("doc_type"):
        deler.append(payload["doc_type"])
    if payload.get("person"):
        deler.append(payload["person"])
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
        pages = extract_with_azure(tmp_path)
    finally:
        os.unlink(tmp_path)

    # To granularitetsnivåer
    fine_chunks = build_chunks(pages, MAX_WORDS_FIN, "fin")
    grov_chunks = build_chunks(pages, MAX_WORDS_GROV, "grov")
    all_chunks = fine_chunks + grov_chunks

    points = []
    for chunk in all_chunks:
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
                "granularitet": chunk["granularitet"],
            }
        ))

    qdrant.upsert(collection_name=COLLECTION, points=points)
    return {
        "status": "ok",
        "chunks_fin": len(fine_chunks),
        "chunks_grov": len(grov_chunks),
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

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    # Søk kun i fine chunks for presisjon
    conditions = [FieldCondition(key="granularitet", match=MatchValue(value="fin"))]
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


# --- Chat ---

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

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    conditions = [FieldCondition(key="granularitet", match=MatchValue(value="fin"))]
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
Svar presist og kildebasert. Oppgi alltid filnavn og sidetall for hver påstand du baserer deg på.
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
