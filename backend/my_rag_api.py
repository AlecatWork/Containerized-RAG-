import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import chromadb
import requests
import os

from config import settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")
logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)


app = FastAPI(title="RAG API", version="1.0", debug=settings.DEBUG)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=False, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

logger.info("Starting RAG API with model %s", settings.MODEL_NAME)
logger.debug("Full configuration: %s", settings.as_dict())


# --- chromaDB --- 
client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
collection = client.get_or_create_collection("documents")


# --- schemas ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    n_results: int = Field(settings.MAX_RESULTS, ge=1, le=20)
    max_distance: float = Field(settings.MAX_DISTANCE, gt=0)

    @field_validator("question")
    @classmethod
    def not_bank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()

class SourceChunk(BaseModel):
    text: str
    source: str
    distance: float

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str


class IngestResponse(BaseModel):
    message: str
    chunks_ingested: int

class StatsResponse(BaseModel):
    document_count: int
    ollama_model: str
    db_path: str


class HealthResponse(BaseModel):
    chromadb: str
    ollama: str
    documents: int


# --- RAG ---
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer ONLY using the provided context. "
    "If the context doesn't contain the answer, say you don't have enough information. "
    "Do not add facts, links, authors, or dates that are not in the context. "
    "Cite sources only by the file name shown in [Source: ...], like (Source: genre.txt). "
    "Keep responses under 100 words."
)

def retrieve(question: str, n_results: int, max_distance: float) -> list[SourceChunk]:
    total = collection.count()
    if total == 0:
        return []
    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, total),
    )

    chunks = []
    for i in range(len(results['documents'][0])):
        distance = results["distances"][0][i]
        if distance <= max_distance:
            chunks.append(SourceChunk(
                text=results["documents"][0][i],
                source= results["metadatas"][0][i].get('source', 'unknown'),
                distance= round(distance, 4),
            ))
    return chunks


def get_confidence(chunks: list[SourceChunk]) -> str:
    if not chunks:
        return "low"
    best = min(c.distance for c in chunks)
    if best < settings.CONFIDENCE_THRESHOLD:
        return "high"
    elif best < settings.MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"
    
    
def generate(messages: list[dict]) -> str:
    try:
        response = requests.post(
            f"{settings.OLLAMA_URL}/api/chat", 
            json={
                "model": settings.MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {"temperature": settings.TEMPERATURE},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
        
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise HTTPException(status_code=503, detail= "Ollama is not running or not responding")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama returned an error: {e}")
    except (KeyError, ValueError):
        raise HTTPException(status_code=502, detail="Unexpected response from Ollama")




# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "RAG API is running", "docs": "/docs"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if collection.count() == 0:
        return AskResponse(
            answer="No documents have been ingested yet. Call POST /ingest first.",
            sources=[],
            confidence="low",
        )
    chunks = retrieve(req.question, req.n_results,req.max_distance)

    if not chunks:
        return AskResponse(
            answer="I don't have relevant information to answer that.",
            sources=[], 
            confidence="low"
        )
    
    confidence = get_confidence(chunks)
    context = "\n\n".join(
         f"[Source: {c.source}]\n{c.text}" for c in chunks
    )
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"},
        {"role": "user", "content": req.question}
    ]

    answer = generate(messages)
    return AskResponse(answer=answer, sources=chunks, confidence=confidence)
    

@app.post("/ingest", response_model=IngestResponse)
def ingest_documents():
    docs_dir = settings.DOCS_DIR
    if not os.path.exists(docs_dir):
        raise HTTPException(status_code=404, detail=f"{docs_dir}/directory ot found")
    
    chunks, ids, metadatas = [], [], []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith((".txt", ".md")):
            continue
        with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8') as f:
            content = f.read()
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append(para)
            ids.append(f"{filename}_{i}")
            metadatas.append({"source": filename, "chunk_index": i})

    if not chunks:
        raise HTTPException(status_code=400, detail=f"No .txt or .md content found in {docs_dir}/")

    
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    return IngestResponse(
        message=f"Ingested {len(chunks)} chunks from {docs_dir}/",
        chunks_ingested=len(chunks),
    ) 


@app.get("/stats", response_model=StatsResponse)
def stats():
    return StatsResponse (
        document_count= collection.count(),
        ollama_model= settings.MODEL_NAME,
        db_path= settings.CHROMA_PATH,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    chroma_status, doc_count = "ok", 0
    try:
        doc_count = collection.count()
    except Exception:
        chroma_status = "unavailable"

    try:
        r = requests.get(f"{settings.OLLAMA_URL}/api/tags", timeout=2)
        ollama_status = "connected" if r.status_code == 200 else "unavailable"
    except requests.exceptions.RequestException:
        ollama_status = "unavailable"

    return HealthResponse (
        chromadb=chroma_status,
        ollama=ollama_status,
        documents=doc_count
    )




