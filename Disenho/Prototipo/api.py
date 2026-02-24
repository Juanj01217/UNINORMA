"""
FastAPI backend para el asistente de normatividad Uninorte.

Expone el pipeline RAG como API REST para ser consumida por el frontend Next.js.

Uso:
    python api.py
    uvicorn api:app --reload --port 8000
"""
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from config import SLM_MODELS, DEFAULT_SLM_MODEL, DEFAULT_EMBEDDING_MODEL
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store, get_retriever
from src.rag_chain import create_rag_chain, query_rag, format_response_with_sources
from src.ollama_client import check_ollama_running, get_available_models

app = FastAPI(
    title="Asistente de Normatividad Uninorte",
    description="API REST para consultar la normatividad institucional de la Universidad del Norte usando RAG + SLM local.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global (se inicializa al arrancar)
_retriever = None
_current_chain = None
_current_model: Optional[str] = None


def _init_retriever():
    global _retriever
    if _retriever is None:
        embedding_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        vector_store = load_vector_store(embedding_model)
        _retriever = get_retriever(vector_store)
    return _retriever


def _init_chain(model_name: str = DEFAULT_SLM_MODEL):
    global _current_chain, _current_model
    if _current_chain is None or _current_model != model_name:
        retriever = _init_retriever()
        _current_chain = create_rag_chain(retriever, model_name)
        _current_model = model_name
    return _current_chain


# --- Schemas ---

class QueryRequest(BaseModel):
    question: str
    model: str = DEFAULT_SLM_MODEL

class SourceInfo(BaseModel):
    source: str
    title: str
    page: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    model: str

class ModelLoadRequest(BaseModel):
    model: str

class StatusResponse(BaseModel):
    ollama_running: bool
    available_models: dict
    active_model: Optional[str]
    vector_store_ready: bool


# --- Endpoints ---

@app.get("/health", response_model=StatusResponse)
def health():
    """Verifica el estado del sistema."""
    ollama_ok = check_ollama_running()
    available = get_available_models() if ollama_ok else {}
    return StatusResponse(
        ollama_running=ollama_ok,
        available_models=available,
        active_model=_current_model,
        vector_store_ready=_retriever is not None,
    )


@app.get("/models")
def list_models():
    """Devuelve la lista de modelos SLM configurados y cuales estan instalados."""
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama no esta activo.")
    available = get_available_models()
    return {
        "models": SLM_MODELS,
        "installed": {m: v for m, v in available.items() if v},
        "active": _current_model,
        "default": DEFAULT_SLM_MODEL,
    }


@app.post("/models/load")
def load_model(body: ModelLoadRequest):
    """Carga (o cambia) el modelo SLM activo."""
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama no esta activo.")
    if body.model not in SLM_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{body.model}' no reconocido.")
    try:
        _init_chain(body.model)
        return {"status": "ok", "model": body.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    """Realiza una consulta RAG sobre la normatividad de Uninorte."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama no esta activo.")

    try:
        chain = _init_chain(body.model)
        result = query_rag(chain, body.question, body.model)
        sources = [
            SourceInfo(
                source=s.get("source", ""),
                title=s.get("title", s.get("source", "")),
                page=str(s.get("page", "")),
            )
            for s in result.get("sources_info", [])
        ]
        # Deduplicar fuentes
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s.source, s.page)
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return QueryResponse(
            answer=result["answer"],
            sources=unique_sources,
            model=result.get("model", body.model),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
