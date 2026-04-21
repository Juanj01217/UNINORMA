"""Configuracion centralizada del prototipo RAG."""
import os
from pathlib import Path

# === Rutas ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"

# === Fuente de PDFs (relativa a la raiz del repo) ===
SCRAPING_PDF_DIR = PROJECT_ROOT.parent / "WebScraping" / "reglamentos"

# === Parametros de Chunking ===
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# === Modelos de Embedding ===
EMBEDDING_MODELS = {
    "minilm-multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet-multilingual": "paraphrase-multilingual-mpnet-base-v2",
}
DEFAULT_EMBEDDING_MODEL = "minilm-multilingual"

# === Configuracion de Ollama ===
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
SLM_MODELS = [
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "llama3.2:1b",
    "llama3.2:3b",
    "phi3:mini",
    "mistral:7b",
    "gemma3:1b",
]
DEFAULT_SLM_MODEL = "qwen2.5:3b"

# === Parametros de Recuperacion ===
RETRIEVAL_TOP_K = 6

# === Parametros de Generacion ===
TEMPERATURE = 0.1
MAX_TOKENS = 2048
