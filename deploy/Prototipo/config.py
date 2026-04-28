"""Configuracion centralizada del prototipo RAG."""
import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# === Rutas ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"

# === Fuente de PDFs (relativa a la raiz del repo) ===
SCRAPING_PDF_DIR = PROJECT_ROOT.parent / "WebScraping" / "reglamentos"

# === Parametros de Chunking ===
# Chunking jerarquico: cada articulo del reglamento es la unidad minima.
# Solo se subdivide si excede ARTICLE_MAX_CHARS. CHUNK_SIZE/CHUNK_OVERLAP se
# mantienen como fallback para documentos sin estructura de articulos.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
ARTICLE_MAX_CHARS = 1500
ARTICLE_MIN_CHARS = 80

# === Modelos de Embedding ===
EMBEDDING_MODELS = {
    "minilm-multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet-multilingual": "paraphrase-multilingual-mpnet-base-v2",
}
DEFAULT_EMBEDDING_MODEL = "minilm-multilingual"

# Backend del embedder. "st" = sentence-transformers (PyTorch, default).
# "onnx" = onnxruntime con modelo cuantizado int8. ONNX baja ~700 MB de RAM
# y elimina la dependencia de torch en runtime; pensado para ARM/Orange Pi.
EMBEDDER_BACKEND = os.environ.get("EMBEDDER_BACKEND", "st").lower()

# === Reranker (cross-encoder) ===
# Modelo cross-encoder multilingue que reordena los chunks recuperados.
# Multiplica el retrieval_accuracy y permite reducir top_k post-rerank,
# compactando el contexto que ve el SLM y bajando latencia de generacion.
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# RERANKER_ENABLED: en hardware limitado (Orange Pi) conviene desactivarlo
# (anade 1-3s por consulta en CPU ARM). Se controla por env var.
RERANKER_ENABLED = _env_bool("RERANKER_ENABLED", True)
RERANKER_TOP_N = 3  # cuantos chunks pasan al prompt final despues del rerank

# === Configuracion de Ollama / SLM responder ===
# Backend del responder. "ollama" (default, cualquier arquitectura) o "rkllm"
# (NPU del Rockchip RK3588 en Orange Pi 5/Pro/Plus). Ver src/llm_backend.py.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
SLM_MODELS = [
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "llama3.2:1b",
    "llama3.2:3b",
    "phi3:mini",
    "gemma3:1b",
    "mistral:7b",
    "llama3.1:8b",
]
DEFAULT_SLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:1.5b")

# Ruta al modelo .rkllm cuando LLM_BACKEND=rkllm. Ignorado en otros backends.
RKLLM_MODEL_PATH = os.environ.get(
    "RKLLM_MODEL_PATH", "/app/models/qwen2.5-1.5b-instruct.rkllm"
)
RKLLM_TARGET_PLATFORM = os.environ.get("RKLLM_TARGET_PLATFORM", "rk3588")

# Modelo dedicado (mas pequeno) para query rewriting. Cualquier modelo de 0.5-1.5B
# rinde bien para una tarea tan acotada y corta la doble llamada LLM a la mitad
# en terminos de tiempo.
REWRITE_SLM_MODEL = "qwen2.5:1.5b"

# keep_alive: tiempo que Ollama mantiene el modelo cargado tras inactividad.
# Sin este valor, el primer query tras 5 min sufre cold-start de 5-15s.
OLLAMA_KEEP_ALIVE = "30m"

# === Parametros de Recuperacion ===
RETRIEVAL_TOP_K = 6
RETRIEVAL_SCORE_THRESHOLD = 0.4

# === Parametros de Generacion ===
TEMPERATURE = 0.1
# Reducido de 600 -> 300. El system prompt limita a 5 oraciones, ~200 tokens
# en espanol. 300 deja margen y recorta la latencia de generacion hasta 50%
# cuando el modelo es verboso.
MAX_TOKENS = 300
