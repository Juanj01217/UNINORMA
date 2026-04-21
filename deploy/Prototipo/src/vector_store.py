"""Gestion del vector store ChromaDB."""
import sys
import shutil
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD

COLLECTION_NAME = "uninorte_normatividad"


def create_vector_store(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    Crea un nuevo vector store ChromaDB a partir de chunks de documentos.

    Sobreescribe la coleccion existente si la hay.
    """
    persist_dir = str(persist_directory)

    # Limpiar directorio existente
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    print(f"Creando vector store con {len(documents)} chunks...")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    print(f"Vector store creado y persistido en {persist_dir}")
    return vector_store


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Carga un vector store ChromaDB existente desde disco."""
    persist_dir = str(persist_directory)

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"No se encontro el vector store en {persist_dir}. "
            f"Ejecuta 'python ingest.py' primero."
        )

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )

    count = vector_store._collection.count()
    print(f"Vector store cargado: {count} chunks indexados")
    return vector_store


def get_retriever(
    vector_store: Chroma,
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
):
    """
    Retriever con umbral de relevancia: solo devuelve chunks cuya similitud coseno
    supere score_threshold (0-1). Si ninguno supera el umbral, devuelve lista vacia,
    lo que permite al RAGChain responder 'no hay informacion' sin llamar al LLM.
    """
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": score_threshold},
    )
