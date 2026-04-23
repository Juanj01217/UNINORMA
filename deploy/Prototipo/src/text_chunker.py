"""Division de documentos en chunks para embedding vectorial.

Estrategia jerarquica (nueva):
  1. Intentar detectar articulos ("Articulo N.", "Art. N.", "ARTICULO N") y usar
     cada articulo como unidad minima de chunk. Preserva la semantica legal:
     una respuesta que cita "Art. 70" no llega al LLM partida por la mitad.
  2. Si el articulo excede ARTICLE_MAX_CHARS, se subdivide con el splitter
     recursivo manteniendo overlap.
  3. Fallback: si no se detectan articulos en el texto (ej. bienestar, informes),
     se aplica el splitter recursivo clasico.
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Agregar raiz del prototipo al path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    ARTICLE_MAX_CHARS,
    ARTICLE_MIN_CHARS,
)


# Regex que detecta el inicio de un articulo. Cubre las variantes mas comunes
# en reglamentos colombianos: "Articulo 12.", "ARTICULO 12.", "Art. 12.",
# opcionalmente con tilde. Captura el numero para metadata.
_ARTICLE_RE = re.compile(
    r"(?im)^\s*(?:art[íi]culo|art\.)\s*(\d+)[\.\-\s]",
)


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


def _find_article_boundaries(text: str) -> List[Tuple[int, int, str]]:
    """Devuelve [(start, end, article_number)] con los limites de cada articulo."""
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return []
    boundaries: List[Tuple[int, int, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        boundaries.append((start, end, m.group(1)))
    return boundaries


def _chunk_by_article(
    page_text: str,
    splitter: RecursiveCharacterTextSplitter,
    max_chars: int = ARTICLE_MAX_CHARS,
    min_chars: int = ARTICLE_MIN_CHARS,
) -> List[Dict[str, Any]]:
    """Divide una pagina respetando limites de articulos.

    Returns: lista de {"text": str, "article": str | None}
    """
    boundaries = _find_article_boundaries(page_text)
    if not boundaries:
        return [{"text": c, "article": None} for c in splitter.split_text(page_text)]

    chunks: List[Dict[str, Any]] = []
    preamble = page_text[: boundaries[0][0]].strip()
    if len(preamble) >= min_chars:
        chunks.append({"text": preamble, "article": None})

    for start, end, art_num in boundaries:
        article_text = page_text[start:end].strip()
        if len(article_text) < min_chars:
            continue
        if len(article_text) <= max_chars:
            chunks.append({"text": article_text, "article": art_num})
        else:
            # Articulo muy largo: subdividir pero etiquetar todos los sub-chunks
            # con el mismo numero de articulo para mantener trazabilidad.
            sub_chunks = splitter.split_text(article_text)
            for sc in sub_chunks:
                chunks.append({"text": sc, "article": art_num})
    return chunks


def chunk_document(
    doc_data: Dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """Divide el texto de un documento en chunks de LangChain.

    Cada chunk incluye metadata con: source, title, page, chunk_index y
    opcionalmente article (numero de articulo detectado).
    """
    chunks = []
    chunk_index = 0

    for page_info in doc_data["pages"]:
        page_text = page_info["text"]
        if not page_text.strip():
            continue

        page_chunks = _chunk_by_article(page_text, splitter)

        for ch in page_chunks:
            metadata = {
                "source": doc_data["filename"],
                "title": doc_data["title"],
                "page": page_info["page_number"],
                "chunk_index": chunk_index,
            }
            if ch["article"] is not None:
                metadata["article"] = ch["article"]

            chunks.append(Document(page_content=ch["text"], metadata=metadata))
            chunk_index += 1

    return chunks


def chunk_all_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Divide todos los documentos extraidos en chunks."""
    splitter = create_splitter(chunk_size, chunk_overlap)
    all_chunks = []
    total_articles = 0

    for doc_data in documents:
        doc_chunks = chunk_document(doc_data, splitter)
        article_chunks = sum(1 for c in doc_chunks if c.metadata.get("article"))
        total_articles += article_chunks
        all_chunks.extend(doc_chunks)
        print(
            f"  {doc_data['filename']}: {len(doc_chunks)} chunks "
            f"({article_chunks} por articulo)"
        )

    print(
        f"\nTotal: {len(all_chunks)} chunks de {len(documents)} documentos "
        f"({total_articles} anclados a articulo)"
    )
    return all_chunks
