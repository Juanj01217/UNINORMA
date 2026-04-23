"""
Reranker cross-encoder post-retrieval.

Aumenta la precision del retrieval reordenando los top-k chunks que devuelve
ChromaDB con un cross-encoder multilingue. El cross-encoder evalua (query, chunk)
como par conjunto, a diferencia del bi-encoder de embeddings que los codifica
por separado. Esto captura interacciones query-chunk que la similitud coseno
no ve.

Ventajas para UNINORMA:
  - Mejor ordenamiento -> el SLM ve los 3 mejores chunks, no los 6 mas "parecidos".
  - Permite bajar top_k del prompt sin perder calidad (prompt mas corto = inferencia
    mas rapida, que es el cuello de botella en CPU).
  - BAAI/bge-reranker-v2-m3 es multilingue y liviano (~560MB).

Fallback: si el reranker no esta disponible (import, descarga, o desactivado),
devuelve los documentos en el orden original sin romper el pipeline.
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RERANKER_MODEL, RERANKER_ENABLED, RERANKER_TOP_N

_logger = logging.getLogger(__name__)

# Cache de modelo a nivel de modulo para evitar recargar en cada query.
_cached_reranker = None
_reranker_load_failed = False


def get_reranker():
    """
    Carga (y cachea) el cross-encoder. Devuelve None si falla o esta desactivado.

    Se importa sentence_transformers de forma perezosa para no forzar su instalacion
    en entornos donde el reranker se desactive por config.
    """
    global _cached_reranker, _reranker_load_failed

    if not RERANKER_ENABLED:
        return None
    if _reranker_load_failed:
        return None
    if _cached_reranker is not None:
        return _cached_reranker

    try:
        from sentence_transformers import CrossEncoder
        _logger.info("Cargando reranker: %s", RERANKER_MODEL)
        _cached_reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        _logger.info("Reranker cargado.")
        return _cached_reranker
    except Exception as exc:
        _logger.warning(
            "No se pudo cargar el reranker (%s). Se continua sin rerank.", exc
        )
        _reranker_load_failed = True
        return None


def rerank_documents(
    query: str,
    docs: List[Document],
    top_n: int = RERANKER_TOP_N,
    reranker=None,
) -> List[Document]:
    """
    Reordena `docs` segun la relevancia del par (query, doc) y devuelve top_n.

    Si `docs` tiene menos elementos que top_n, devuelve la lista ordenada completa.
    Si no hay reranker disponible, hace un fallback truncando al top_n original.
    """
    if not docs:
        return []

    if reranker is None:
        reranker = get_reranker()

    # Fallback: sin cross-encoder, al menos limita el tamano del contexto
    # siguiendo el orden del retriever (ya ordenado por similitud coseno).
    if reranker is None:
        return docs[:top_n]

    try:
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]
    except Exception as exc:
        _logger.warning("Reranker fallo en runtime (%s). Fallback a orden original.", exc)
        return docs[:top_n]
