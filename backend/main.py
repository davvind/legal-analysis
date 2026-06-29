from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os, tempfile, re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
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
MAX_WORDS_FIN = 400
MAX_WORDS_GROV = 1500
MIN_WORDS = 40


def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )


# ── Dokumentid-deteksjon ──────────────────────────────────────────────────────

def decode_doubled(text: str) -> str:
    """
    Konverterer OCR-artefakt med dobbelttegn til normal tekst.
    'ddookkuummeennttiidd' → 'dokumentid'
    """
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


def extract_footer_dokumentid(page_text_bottom: str) -> Optional[str]:
    """
    Ekstraherer dokumentid fra bunntekst.
    Håndterer både normal tekst og OCR-dobbelttegn-variant.
    """
    decoded = decode_doubled(page_text_bottom)
    match = re.search(r'dokumentid[:\s]*(\d+)', decoded, re.IGNORECASE)
    return match.group(1) if match else None


# ── Azure Document Intelligence ───────────────────────────────────────────────

def extract_with_azure(pdf_path: str) -> list[dict]:
    """
    Ekstraherer tekst per side via Azure DI Read-modellen.
    Returnerer liste av {page_num, text, dokumentid, footer_text}.
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
        lines = [line.content for line in (page.lines or [])]
        if not lines:
            continue

        # Bunntekst: siste 3 linjer brukes for dokumentid-deteksjon
        footer_lines = lines[-3:]
        footer_text = " ".join(footer_lines)
        body_text = "\n".join(lines)

        dokumentid = extract_footer_dokumentid(footer_text)

        pages.append({
            "page_num": page.page_number,
            "text": body_text,
            "footer_text": footer_text,
            "dokumentid": dokumentid,
        })

    return pages


# ── Segmentering i sub-dokumenter ─────────────────────────────────────────────

def segment_into_subdocs(pages: list[dict]) -> list[dict]:
    """
    Grupperer sider i sub-dokumenter basert på dokumentid-skifter.
    Hvert sub-dokument er et sammenhengende originaldokument.
    Returnerer liste av {dokumentid, sider: [{page_num, text}]}.
    """
    subdocs = []
    current_id = None
    current_pages = []

    for page in pages:
        pid = page["dokumentid"]

        # Dokumentskifte: ny dokumentid dukker opp
        if pid and pid != current_id:
            if current_pages:
                subdocs.append({
                    "dokumentid": current_id or "ukjent",
                    "sider": current_pages
                })
            current_id = pid
            current_pages = [page]
        else:
            current_pages.append(page)

    # Siste sub-dokument
    if current_pages:
        subdocs.append({
            "dokumentid": current_id or "ukjent",
            "sider": current_pages
        })

    return subdocs


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_into_paragraphs(text: str) -> list[str]:
    raw = re.split(r'\n{2,}|\n(?=[A-ZÆØÅ0-9])', text)
    return [p.strip() for p in raw if len(p.split()) >= 5]


def build_chunks(pages: list[dict], max_words: int, granularitet: str,
                 prefix_context: str = "") -> list[dict]:
    """
    Avsnittbasert chunking med sidesporring og overlapp.
    prefix_context: siste avsnitt fra forrige fil (cross-file stitching).
    """
    all_paras = []
    for page in pages:
        for p in split_into_paragraphs(page["text"]):
            all_paras.append({"text": p, "page": page["page_num"]})

    if not all_paras:
        return []

    chunks = []
    i = 0

    # Cross-file overlapp: inject kontekst fra forrige del som første para
    if prefix_context:
        last_para = {"text": prefix_context, "page": all_paras[0]["page"]}
    else:
        last_para = None

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
                sentences = re.split(r'(?<=[.!?])\s+', para["text"])
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

        last_para = all_paras[i - 1] if i > 0 and i <= len(all_paras) else None

    return chunks


# ── Cross-file stitching: hent siste chunk fra forrige del ───────────────────

def hent_forrige_del_kontekst(pdf_id: str, del_nummer: int) -> str:
    """
    Henter siste fine chunk fra del_nummer-1 med samme pdf_id fra Qdrant.
    Brukes som overlapp-kontekst ved starten av ny del.
    """
    if del_nummer <= 1:
        return ""

    try:
        results, _ = qdrant.scroll(
            collection_name=COLLECTION,
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

        # Finn chunk med høyest side_slutt
        siste = max(results, key=lambda r: r.payload.get("side_slutt", 0))
        tekst = siste.payload.get("text", "")
        # Returner siste avsnitt av teksten
        avsnitt = [p for p in tekst.split("\n\n") if p.strip()]
        return avsnitt[-1] if avsnitt else ""

    except Exception:
        return ""


# ── Embedding og kildeformat ──────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


def format_kilde(payload: dict) -> str:
    filename = payload.get("filename", "ukjent fil")
    s0 = payload.get("side_start", "?")
    s1 = payload.get("side_slutt", "?")
    sider = f"s. {s0}" if s0 == s1 else f"s. {s0}–{s1}"
    subdok = payload.get("sub_dokumentid")
    deler = [filename, sider]
    if subdok and subdok != "ukjent":
        deler.append(f"dok.id {subdok}")
    if payload.get("doc_type"):
        deler.append(payload["doc_type"])
    if payload.get("person"):
        deler.append(payload["person"])
    return " | ".join(deler)


# ── Endepunkter ───────────────────────────────────────────────────────────────

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
    dato: str = Form(""),
    pdf_id: str = Form(""),        # Felles ID for alle deler av samme original-PDF
    del_nummer: int = Form(1),     # 1-basert rekkefølge innen pdf_id
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

    # Cross-file stitching: hent kontekst fra forrige del
    prefix_context = ""
    if pdf_id and del_nummer > 1:
        prefix_context = hent_forrige_del_kontekst(pdf_id, del_nummer)

    # Segmenter i sub-dokumenter basert på dokumentid
    subdocs = segment_into_subdocs(pages)

    points = []
    chunks_fin_total = 0
    chunks_grov_total = 0

    for subdoc in subdocs:
        sub_id = subdoc["dokumentid"]
        subdoc_pages = subdoc["sider"]

        # Kun første sub-dokument i filen får cross-file prefix
        ctx = prefix_context if subdoc == subdocs[0] else ""

        fine_chunks = build_chunks(subdoc_pages, MAX_WORDS_FIN, "fin", ctx)
        grov_chunks = build_chunks(subdoc_pages, MAX_WORDS_GROV, "grov", ctx)
        chunks_fin_total += len(fine_chunks)
        chunks_grov_total += len(grov_chunks)

        for chunk in fine_chunks + grov_chunks:
            if len(chunk["text"].strip()) < 50:
                continue
            vector = embed(chunk["text"])
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "filename": file.filename,
                    "pdf_id": pdf_id or file.filename,
                    "del_nummer": del_nummer,
                    "sub_dokumentid": sub_id,
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
        "chunks_fin": chunks_fin_total,
        "chunks_grov": chunks_grov_total,
        "sub_dokumenter": len(subdocs),
        "sider_prosessert": len(pages),
        "filename": file.filename,
        "cross_file_stitching": bool(prefix_context),
    }


@app.post("/analyze")
async def analyze(
    query: str = Form(...),
    analyse_type: str = Form("general"),
    filter_person: str = Form(""),
    filter_doctype: str = Form("")
):
    query_vector = embed(query)

    conditions = [FieldCondition(key="granularitet", match=MatchValue(value="fin"))]
    if filter_person:
        conditions.append(FieldCondition(key="person", match=MatchValue(value=filter_person)))
    if filter_doctype:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=filter_doctype)))

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=15,
        query_filter=Filter(must=conditions)
    )

    context = "\n\n---\n\n".join([
        f"[{format_kilde(r.payload)}]\n{r.payload['text']}"
        for r in results
    ])

    prompts = {
        "motsetninger": f"""Analyser følgende kildemateriale og identifiser motstridende forklaringer eller påstander om: {query}

Vær konkret: (1) Hva påstår kilde A? (2) Hva påstår kilde B? (3) Hva er den faktiske motsetningen?
Oppgi alltid filnavn, dokumentid og sidetall for hver påstand.

KILDEMATERIALE:
{context}""",
        "tidslinje": f"""Sett opp en kronologisk analyse av alle hendelser relatert til: {query}

Marker eksplisitt hvor forklaringer eller bevis ikke stemmer med tidslinjen.
Oppgi alltid filnavn, dokumentid og sidetall for hver hendelse.

KILDEMATERIALE:
{context}""",
        "selvmotsigelser": f"""Analyser om samme person/kilde motsier seg selv angående: {query}

Vis konkret hva som ble sagt, i hvilket dokument og på hvilken side, og hva som er selvmotsigende.

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

    conditions = [FieldCondition(key="granularitet", match=MatchValue(value="fin"))]
    if req.filter_person:
        conditions.append(FieldCondition(key="person", match=MatchValue(value=req.filter_person)))
    if req.filter_doctype:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=req.filter_doctype)))

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=12,
        query_filter=Filter(must=conditions)
    )

    context = "\n\n---\n\n".join([
        f"[{format_kilde(r.payload)}]\n{r.payload['text']}"
        for r in results
    ])

    system_prompt = f"""Du er en juridisk analytiker som jobber med norske rettsdokumenter.
Svar presist og kildebasert. For hver påstand du baserer deg på, oppgi:
- filnavn
- dokumentid (f.eks. "dok.id 83654328")
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
        "kilder_brukt": len(results),
        "query": latest_query
    }
