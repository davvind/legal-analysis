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

COLLECTION_KORPUS = "legal_docs"
COLLECTION_EGET   = "egenproduksjon"
EMBED_MODEL       = "text-embedding-3-large"
EMBED_DIM         = 3072
MAX_WORDS_FIN     = 400    # Korpus: fin
MAX_WORDS_GROV    = 1500   # Korpus: grov (GraphRAG)
MAX_WORDS_EGET    = 250    # Egenproduksjon: avsnitt som enhet
MIN_WORDS         = 40


# ── Collections ───────────────────────────────────────────────────────────────

def ensure_collection(name: str):
    existing = [c.name for c in qdrant.get_collections().collections]
    if name not in existing:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )


# ── Dokumentid-deteksjon (korpus) ─────────────────────────────────────────────

def decode_doubled(text: str) -> str:
    """Konverterer OCR-dobbelttegn-artefakt: 'ddookkuummeennttiidd' → 'dokumentid'"""
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
    """
    Ekstraherer tekst per side via Azure DI prebuilt-read.
    Returnerer liste av {page_num, text, dokumentid}.
    """
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
    """Grupperer sider i sub-dokumenter basert på dokumentid-skifter."""
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
    """
    Chunking for egenproduksjon: avsnitt som naturlig enhet, maks 250 ord.
    Ingen grov-variant. Ingen cross-file stitching.
    """
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


# ── Endepunkter ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Legal Analysis API kjører"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_korpus(
    file: UploadFile = File(...),
    doc_type: str = Form("ukjent"),
    person: str = Form(""),
    dato: str = Form(""),
    pdf_id: str = Form(""),
    del_nummer: int = Form(1),
):
    """Laster opp korpusdokument med full segmentering og to granularitetsnivåer."""
    ensure_collection(COLLECTION_KORPUS)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pages = extract_with_azure(tmp_path)
    finally:
        os.unlink(tmp_path)

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
    }


@app.post("/upload/egen")
async def upload_egen(
    file: UploadFile = File(...),
    doc_type: str = Form("egenproduksjon"),
    person: str = Form(""),
    dato: str = Form(""),
    tittel: str = Form(""),
):
    """Laster opp egenprodusert dokument med avsnittbasert chunking (maks 250 ord)."""
    ensure_collection(COLLECTION_EGET)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pages = extract_with_azure(tmp_path)
    finally:
        os.unlink(tmp_path)

    chunks = build_chunks_eget(pages)
    points = []

    for chunk in chunks:
        if len(chunk["text"].strip()) < 30:
            continue
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
            }
        ))

    qdrant.upsert(collection_name=COLLECTION_EGET, points=points)

    return {
        "status": "ok",
        "chunks_indeksert": len(points),
        "sider_prosessert": len(pages),
        "filename": file.filename,
    }


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
        eget_results = qdrant.search(
            collection_name=COLLECTION_EGET,
            query_vector=query_vector,
            limit=5,
        )
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
        eget_results = qdrant.search(
            collection_name=COLLECTION_EGET,
            query_vector=query_vector,
            limit=4,
        )
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
