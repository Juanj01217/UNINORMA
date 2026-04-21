"""
FastAPI backend para el asistente de normatividad Uninorte.

Expone el pipeline RAG como API REST para ser consumida por el frontend Next.js.

Uso:
    python api.py
    uvicorn api:app --reload --port 8000
"""
import json
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from config import SLM_MODELS, DEFAULT_SLM_MODEL, DEFAULT_EMBEDDING_MODEL
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store, get_retriever
from src.rag_chain import create_rag_chain, query_rag, format_response_with_sources
from src.ollama_client import check_ollama_running, get_available_models

_OLLAMA_NO_ACTIVO = "Ollama no esta activo."

app = FastAPI(
    title="Asistente de Normatividad Uninorte",
    description="API REST para consultar la normatividad institucional de la Universidad del Norte usando RAG + SLM local.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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


@app.on_event("startup")
async def startup_event():
    """Inicializa el vector store al arrancar para que /health reporte correctamente."""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_retriever)


def _init_chain(model_name: str = DEFAULT_SLM_MODEL):
    global _current_chain, _current_model
    if _current_chain is None or _current_model != model_name:
        retriever = _init_retriever()
        _current_chain = create_rag_chain(retriever, model_name)
        _current_model = model_name
    return _current_chain


# --- Schemas ---

class HistoryMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class QueryRequest(BaseModel):
    question: str
    model: str = DEFAULT_SLM_MODEL
    history: list[HistoryMessage] = []

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
    active_model: Optional[str] = None
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


@app.get("/models", responses={503: {"description": "Ollama no esta activo."}})
def list_models():
    """Devuelve la lista de modelos SLM configurados y cuales estan instalados."""
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail=_OLLAMA_NO_ACTIVO)
    available = get_available_models()
    installed_models = [m for m in SLM_MODELS if available.get(m, False)]
    return {
        "models": installed_models,
        "installed": {m: v for m, v in available.items() if v},
        "active": _current_model,
        "default": DEFAULT_SLM_MODEL,
    }


@app.post("/models/load", responses={
    400: {"description": "Modelo no reconocido."},
    500: {"description": "Error interno al cargar el modelo."},
    503: {"description": "Ollama no esta activo."},
})
def load_model(body: ModelLoadRequest):
    """Carga (o cambia) el modelo SLM activo."""
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail=_OLLAMA_NO_ACTIVO)
    if body.model not in SLM_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{body.model}' no reconocido.")
    try:
        _init_chain(body.model)
        return {"status": "ok", "model": body.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, responses={
    400: {"description": "Pregunta vacia."},
    500: {"description": "Error en el pipeline RAG."},
    503: {"description": "Ollama no esta activo."},
})
def query(body: QueryRequest):
    """Realiza una consulta RAG sobre la normatividad de Uninorte."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail=_OLLAMA_NO_ACTIVO)

    try:
        chain = _init_chain(body.model)
        history_dicts = [{"role": h.role, "content": h.content} for h in body.history]
        result = query_rag(chain, body.question, body.model, history=history_dicts)
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


@app.post("/query/stream", responses={
    400: {"description": "Pregunta vacia."},
    503: {"description": "Ollama no esta activo."},
})
def query_stream(body: QueryRequest):
    """Consulta RAG con respuesta en streaming (Server-Sent Events)."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")
    if not check_ollama_running():
        raise HTTPException(status_code=503, detail=_OLLAMA_NO_ACTIVO)

    chain = _init_chain(body.model)
    history_dicts = [{"role": h.role, "content": h.content} for h in body.history]
    sources_info, _, token_stream = chain.invoke_stream(body.question, history=history_dicts)

    def generate():
        try:
            accumulated = ""
            for chunk in token_stream:
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    accumulated += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
            # Si el LLM decidio por su cuenta que no habia informacion, no mostrar fuentes
            llm_said_no_info = "no encontre informacion" in accumulated.lower()[:120]
            final_sources = [] if llm_said_no_info else sources_info
            yield f"data: {json.dumps({'done': True, 'sources': final_sources, 'model': body.model})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

# Almacén en memoria de jobs de benchmark (clave: job_id)
_benchmark_jobs: dict = {}


class BenchmarkStartRequest(BaseModel):
    models: list[str]
    quick: bool = True  # True = 5 preguntas, False = todas (~40)


def _compute_benchmark_summary(results: list) -> dict:
    """Agrega métricas por modelo a partir de los resultados crudos."""
    by_model: dict = {}
    for r in results:
        m = r["model_name"]
        by_model.setdefault(m, []).append(r)

    summary = {}
    for model_name, mrs in by_model.items():
        valid = [r for r in mrs if r["latency_seconds"] >= 0]
        if not valid:
            continue
        summary[model_name] = {
            "total_questions": len(mrs),
            "successful": len(valid),
            "avg_latency_seconds": round(
                sum(r["latency_seconds"] for r in valid) / len(valid), 3
            ),
            "retrieval_accuracy": round(
                sum(1 for r in valid if r["retrieval_hit"]) / len(valid), 3
            ),
            "avg_answer_relevancy": round(
                sum(r["answer_relevancy"] for r in valid) / len(valid), 3
            ),
            "avg_faithfulness": round(
                sum(r["faithfulness"] for r in valid) / len(valid), 3
            ),
            "hallucination_rate": round(
                sum(1 for r in valid if r["hallucination_detected"]) / len(valid), 3
            ),
            "avg_memory_mb": round(
                sum(r["memory_usage_mb"] for r in valid) / len(valid), 2
            ),
        }
    return summary


def _save_benchmark_to_disk(results: list, summary: dict) -> None:
    output_dir = Path(__file__).parent / "benchmark" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"{ts}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(output_dir / f"{ts}_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _sample_questions_quick(questions: list) -> list:
    seen: set = set()
    sampled = []
    for q in questions:
        cat = q.get("category", "")
        if cat not in seen:
            seen.add(cat)
            sampled.append(q)
        if len(sampled) >= 6:
            break
    return sampled


def _eval_one_question(chain, q: dict, model_name: str, raw_sentence_model, metrics) -> dict:
    mem_before = metrics["get_memory_usage_mb"]()
    try:
        rag_result, latency = metrics["measure_latency"](
            metrics["query_rag"], chain, q["question"], model_name
        )
        mem_after = metrics["get_memory_usage_mb"]()
        answer = rag_result["answer"]
        source_docs = rag_result.get("source_documents", [])
        retrieved_sources = [doc.metadata.get("source", "") for doc in source_docs]
        context = "\n".join(doc.page_content for doc in source_docs)
        return {
            "question_id": q["id"],
            "model_name": model_name,
            "question": q["question"],
            "answer": answer,
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "latency_seconds": round(latency, 3),
            "memory_usage_mb": round(max(0.0, mem_after - mem_before), 2),
            "retrieval_hit": metrics["check_retrieval_hit"](retrieved_sources, q["expected_source"]),
            "answer_relevancy": round(metrics["compute_answer_relevancy"](q["question"], answer, raw_sentence_model), 3),
            "faithfulness": round(metrics["compute_faithfulness"](answer, context), 3),
            "hallucination_detected": metrics["detect_hallucination"](answer, context),
            "no_answer_correct": metrics["check_no_answer_correct"](answer, q["expected_source"]),
        }
    except Exception as exc:
        return {
            "question_id": q["id"],
            "model_name": model_name,
            "question": q["question"],
            "answer": f"ERROR: {exc}",
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "latency_seconds": -1,
            "memory_usage_mb": 0.0,
            "retrieval_hit": False,
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "hallucination_detected": False,
            "no_answer_correct": False,
        }


def _run_benchmark_thread(job_id: str, models: list, quick: bool) -> None:
    """Ejecuta el benchmark en un hilo separado y actualiza el job en memoria."""
    job = _benchmark_jobs[job_id]
    try:
        from config import EMBEDDING_MODELS
        from sentence_transformers import SentenceTransformer
        from src.embeddings import get_embedding_model
        from src.rag_chain import create_rag_chain, query_rag
        from src.vector_store import get_retriever, load_vector_store
        from benchmark.metrics import (
            check_no_answer_correct, check_retrieval_hit,
            compute_answer_relevancy, compute_faithfulness,
            detect_hallucination, get_memory_usage_mb, measure_latency,
        )

        questions_path = Path(__file__).parent / "benchmark" / "test_questions.json"
        with open(questions_path, "r", encoding="utf-8") as f:
            questions: list = json.load(f)["questions"]

        if quick:
            questions = _sample_questions_quick(questions)

        job["total_questions"] = len(questions) * len(models)

        emb_model_obj = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        vector_store = load_vector_store(emb_model_obj)
        retriever = get_retriever(vector_store)
        raw_sentence_model = SentenceTransformer(EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL])

        metrics = {
            "query_rag": query_rag,
            "measure_latency": measure_latency,
            "get_memory_usage_mb": get_memory_usage_mb,
            "check_retrieval_hit": check_retrieval_hit,
            "compute_answer_relevancy": compute_answer_relevancy,
            "compute_faithfulness": compute_faithfulness,
            "detect_hallucination": detect_hallucination,
            "check_no_answer_correct": check_no_answer_correct,
        }

        all_results: list = []
        for model_name in models:
            job["current_model"] = model_name
            chain = create_rag_chain(retriever, model_name)
            for q in questions:
                result = _eval_one_question(chain, q, model_name, raw_sentence_model, metrics)
                all_results.append(result)
                job["completed_questions"] += 1
                job["results"] = all_results
                job["summary"] = _compute_benchmark_summary(all_results)

        job["status"] = "done"
        job["current_model"] = None
        _save_benchmark_to_disk(all_results, job["summary"])

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/benchmark/start", responses={
    400: {"description": "Modelos no reconocidos o lista vacia."},
})
def benchmark_start(body: BenchmarkStartRequest):
    """Inicia un benchmark en background y devuelve el job_id para hacer polling."""
    invalid = [m for m in body.models if m not in SLM_MODELS]
    if invalid:
        raise HTTPException(
            status_code=400, detail=f"Modelos no reconocidos: {invalid}"
        )
    if not body.models:
        raise HTTPException(status_code=400, detail="Selecciona al menos un modelo.")

    job_id = str(uuid.uuid4())[:8]
    _benchmark_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "models": body.models,
        "quick": body.quick,
        "total_questions": 0,
        "completed_questions": 0,
        "current_model": None,
        "results": [],
        "summary": {},
        "error": None,
    }

    thread = threading.Thread(
        target=_run_benchmark_thread,
        args=(job_id, body.models, body.quick),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/benchmark/progress/{job_id}", responses={
    404: {"description": "Job no encontrado."},
})
def benchmark_progress(job_id: str):
    """Retorna el estado y progreso de un job de benchmark (para polling)."""
    if job_id not in _benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    return _benchmark_jobs[job_id]


@app.get("/benchmark/results")
def benchmark_results():
    """Retorna los últimos resultados guardados en disco (máx. 10 runs)."""
    results_dir = Path(__file__).parent / "benchmark" / "results"
    if not results_dir.exists():
        return {"runs": []}

    runs = []
    for f in sorted(results_dir.glob("*_summary.json"), reverse=True)[:10]:
        try:
            with open(f, encoding="utf-8") as fp:
                summary = json.load(fp)
            timestamp_str = f.stem.replace("_summary", "")
            runs.append({"timestamp": timestamp_str, "summary": summary})
        except Exception:
            pass

    return {"runs": runs}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("api:app", host=host, port=8000, reload=True)
