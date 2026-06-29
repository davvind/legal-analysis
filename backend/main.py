from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import os, tempfile
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import pymupdf4llm
import uuid

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
        text = pymupdf4llm.to_markdown(tmp_path)
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
