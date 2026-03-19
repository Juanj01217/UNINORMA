"""Division de documentos en chunks para embedding vectorial."""
import sys
from pathlib import Path
from typing import Dict, List, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Agregar raiz del prototipo al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS


def create_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: List[str] = None,
) -> RecursiveCharacterTextSplitter:
    """Crea un text splitter configurado."""
    if separators is None:
        separators = SEPARATORS
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )


def chunk_document(
    doc_data: Dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """
    Divide el texto de un documento en chunks de LangChain.

    Cada chunk incluye metadata con: source, title, page, chunk_index.
    """
    chunks = []
    chunk_index = 0

    for page_info in doc_data["pages"]:
        page_text = page_info["text"]
        if not page_text.strip():
            continue

        page_chunks = splitter.split_text(page_text)

        for chunk_text in page_chunks:
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": doc_data["filename"],
                    "title": doc_data["title"],
                    "page": page_info["page_number"],
                    "chunk_index": chunk_index,
                },
            )
            chunks.append(doc)
            chunk_index += 1

    return chunks


def chunk_all_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Divide todos los documentos extraidos en chunks.

    Returns:
        Lista plana de todos los chunks Document de todos los PDFs.
    """
    splitter = create_splitter(chunk_size, chunk_overlap)
    all_chunks = []

    for doc_data in documents:
        doc_chunks = chunk_document(doc_data, splitter)
        all_chunks.extend(doc_chunks)
        print(f"  {doc_data['filename']}: {len(doc_chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks de {len(documents)} documentos")
    return all_chunks
