"""Gestion de modelos de embedding multilingue."""
import sys
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL


def get_embedding_model(
    model_key: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """
    Carga un modelo de embedding HuggingFace compatible con LangChain.

    Args:
        model_key: Clave del diccionario EMBEDDING_MODELS en config.

    Returns:
        HuggingFaceEmbeddings listo para usar con ChromaDB.
    """
    model_name = EMBEDDING_MODELS[model_key]
    print(f"Cargando modelo de embedding: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Modelo de embedding cargado exitosamente")
    return embeddings
