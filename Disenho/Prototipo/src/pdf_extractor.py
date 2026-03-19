"""Extraccion de texto de documentos PDF usando PyMuPDF."""
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import unquote


def clean_text(raw_text: str) -> str:
    """Limpia texto extraido de PDF: normaliza espacios, encoding, etc."""
    text = raw_text
    # Normalizar saltos de linea multiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Eliminar espacios multiples (pero no saltos de linea)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Eliminar lineas que solo tienen numeros (paginacion)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Limpiar espacios al inicio/final de lineas
    text = re.sub(r"^ +", "", text, flags=re.MULTILINE)
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    # Eliminar lineas vacias consecutivas
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_filename(filename: str) -> str:
    """Convierte un nombre de archivo URL-encoded a titulo legible."""
    name = unquote(filename)
    name = name.replace(".pdf", "").replace("_", " ").replace("%20", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extrae todo el texto de un archivo PDF.

    Returns:
        Diccionario con filename, title, num_pages, full_text, pages.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    full_text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        cleaned = clean_text(text)
        pages.append({
            "page_number": page_num + 1,
            "text": cleaned,
        })
        full_text_parts.append(cleaned)

    doc.close()

    filename = pdf_path.name
    full_text = "\n\n".join(full_text_parts)

    return {
        "filename": filename,
        "title": clean_filename(filename),
        "num_pages": len(pages),
        "full_text": full_text,
        "pages": pages,
    }


def extract_all_pdfs(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Extrae texto de todos los PDFs en un directorio."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron PDFs en {pdf_dir}")

    documents = []
    for pdf_path in pdf_files:
        print(f"  Extrayendo: {pdf_path.name}")
        try:
            doc_data = extract_text_from_pdf(pdf_path)
            if doc_data["full_text"].strip():
                documents.append(doc_data)
                print(f"    -> {doc_data['num_pages']} paginas, "
                      f"{len(doc_data['full_text'])} caracteres")
            else:
                print("    -> ADVERTENCIA: Sin texto extraible")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    return documents


def save_processed_text(doc_data: Dict[str, Any], output_dir: Path) -> Path:
    """Guarda texto extraido y limpio en un archivo .txt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\-.]", "_", doc_data["filename"])
    output_path = output_dir / f"{safe_name}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Fuente: {doc_data['filename']}\n")
        f.write(f"# Titulo: {doc_data['title']}\n")
        f.write(f"# Paginas: {doc_data['num_pages']}\n")
        f.write("=" * 60 + "\n\n")
        f.write(doc_data["full_text"])

    return output_path
