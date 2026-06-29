from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import os, tempfile, json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import pymupdf4llm
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

def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def embed(text):
    response = openai_client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding

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
        import fitz
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    finally:
        os.unlink(tmp_path)

    chunks = chunk_text(text)
    points = []
    for chunk in chunks:
        if len(chunk.strip()) < 50:
            continue
        vector = embed(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk,
                "filename": file.filename,
                "doc_type": doc_type,
                "person": person,
                "dato": dato
            }
        ))

    qdrant.upsert(collection_name=COLLECTION, points=points)
    return {"status": "ok", "chunks_indexed": len(points), "filename": file.filename}

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
        f"[Fil: {r.payload['filename']} | Type: {r.payload['doc_type']} | Person: {r.payload['person']}]\n{r.payload['text']}"
        for r in results
    ])

    prompts = {
        "motsetninger": f"""Analyser følgende kildemateriale og identifiser motstridende forklaringer eller påstander om: {query}
        
Vær konkret: (1) Hva påstår kilde A? (2) Hva påstår kilde B? (3) Hva er den faktiske motsetningen?

KILDEMATERIALE:
{context}""",
        "tidslinje": f"""Sett opp en kronologisk analyse av alle hendelser relatert til: {query}

Marker eksplisitt hvor forklaringer eller bevis ikke stemmer med tidslinjen.

KILDEMATERIALE:
{context}""",
        "selvmotsigelser": f"""Analyser om samme person/kilde motsier seg selv angående: {query}

Vis konkret hva som ble sagt, i hvilket dokument, og hva som er selvmotsigende.

KILDEMATERIALE:
{context}""",
        "general": f"""Analyser følgende kildemateriale og svar på: {query}

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
    role: str  # "user" eller "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    analyse_type: Optional[str] = "general"
    filter_person: Optional[str] = ""
    filter_doctype: Optional[str] = ""

@app.post("/chat")
async def chat(req: ChatRequest):
    # Bruk siste brukermelding som søkespørsmål mot Qdrant
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        return {"error": "Ingen brukermelding funnet"}
    
    latest_query = user_messages[-1].content

    # Hent relevante dokumentbiter basert på siste spørsmål
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
        f"[Fil: {r.payload['filename']} | Type: {r.payload['doc_type']} | Person: {r.payload['person']}]\n{r.payload['text']}"
        for r in results
    ])

    # Systemmelding som setter rollen og injiserer kontekst
    system_prompt = f"""Du er en juridisk analytiker som jobber med norske rettsdokumenter. 
Svar presist og kildebasert. Oppgi alltid hvilke dokumenter påstandene stammer fra.
Analysetypen er: {req.analyse_type}

RELEVANT KILDEMATERIALE (hentet fra dokumentbasen basert på siste spørsmål):
{context}

Hvis et spørsmål ikke kan besvares ut fra kildematerialet, si det eksplisitt."""

    # Bygg full meldingshistorikk til OpenAI
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
