"""
Pipeline de ingesta: Scraping completo -> texto -> chunks -> embeddings -> ChromaDB.

Maneja 3 tipos de contenido de la pagina de normatividad Uninorte:
  1. PDFs directos
  2. Sub-paginas con PDFs (los sigue y descarga)
  3. Paginas web sin PDF (extrae texto HTML)

Uso:
    python ingest.py              (usa cache local o descarga)
    python ingest.py --download   (fuerza re-descarga completa)
    python ingest.py --pdf-dir ruta/a/pdfs  (solo PDFs locales)
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import SCRAPING_PDF_DIR, RAW_PDF_DIR, PROCESSED_DIR, DEFAULT_EMBEDDING_MODEL
from src.pdf_extractor import extract_all_pdfs, save_processed_text
from src.text_chunker import chunk_all_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.web_scraper import scrape_normatividad


def copy_pdfs_to_data(source_dir: Path, dest_dir: Path) -> int:
    """Copia PDFs del directorio de scraping al directorio de datos del prototipo."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(source_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No se encontraron PDFs en {source_dir}.\n"
            f"Opciones:\n"
            f"  1. Ejecuta: python ingest.py --download\n"
            f"  2. Coloca PDFs manualmente en: {dest_dir}"
        )

    for pdf in pdf_files:
        dest = dest_dir / pdf.name
        if not dest.exists():
            shutil.copy2(pdf, dest)
            print(f"  Copiado: {pdf.name}")
        else:
            print(f"  Ya existe: {pdf.name}")

    return len(pdf_files)


def obtain_content(dest_dir: Path, force_download: bool = False):
    """
    Obtiene todo el contenido de normatividad.

    Returns:
        web_documents: Lista de documentos extraidos de paginas web (o lista vacia)
    """
    # Siempre hacer scraping completo para capturar PDFs + contenido web
    # Los PDFs existentes no se re-descargan si ya estan en disco
    print("  Ejecutando scraping completo de normatividad Uninorte...")
    print()

    # Limpiar PDFs duplicados antes de scraping si se fuerza descarga
    if force_download:
        existing_pdfs = list(dest_dir.glob("*.pdf"))
        if existing_pdfs:
            print(f"  Limpiando {len(existing_pdfs)} PDFs existentes para descarga limpia...")
            for pdf in existing_pdfs:
                pdf.unlink()

    results = scrape_normatividad(pdf_dest_dir=dest_dir)

    print(f"\n  Resumen scraping:")
    print(f"    PDFs descargados: {results['pdfs_downloaded']}")
    print(f"    Documentos web extraidos: {len(results['web_documents'])}")
    if results["errors"]:
        print(f"    Errores: {len(results['errors'])}")
        for err in results["errors"]:
            print(f"      - {err}")

    return results["web_documents"]


def run_ingestion(
    pdf_dir: Path = None,
    embedding_model_key: str = DEFAULT_EMBEDDING_MODEL,
    force_download: bool = False,
) -> None:
    """Ejecuta el pipeline completo de ingesta."""

    web_documents = []

    # Paso 1: Obtener contenido
    if pdf_dir is None:
        pdf_dir = RAW_PDF_DIR
        print("=" * 60)
        print("PASO 1: Obteniendo contenido de normatividad")
        print("=" * 60)
        web_documents = obtain_content(RAW_PDF_DIR, force_download)
        pdf_count = len(list(pdf_dir.glob("*.pdf")))
        print(f"-> {pdf_count} PDFs + {len(web_documents)} paginas web\n")
    else:
        print("=" * 60)
        print(f"PASO 1: Usando PDFs de {pdf_dir}")
        print("=" * 60 + "\n")

    # Paso 2: Extraer texto de PDFs
    print("=" * 60)
    print("PASO 2: Extrayendo texto de PDFs")
    print("=" * 60)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if pdf_files:
        documents = extract_all_pdfs(pdf_dir)
        print(f"-> {len(documents)} PDFs procesados\n")
    else:
        documents = []
        print("-> No hay PDFs para procesar\n")

    # Paso 2b: Agregar documentos web
    if web_documents:
        print("=" * 60)
        print("PASO 2b: Procesando paginas web extraidas")
        print("=" * 60)
        for doc in web_documents:
            print(f"  {doc['title']}: {len(doc['full_text'])} chars, {doc['num_pages']} secciones")
        documents.extend(web_documents)
        print(f"-> {len(web_documents)} paginas web agregadas\n")

    if not documents:
        print("ERROR: No hay documentos disponibles para procesar.")
        print("Ejecuta: python ingest.py --download")
        sys.exit(1)

    # Paso 3: Guardar texto procesado
    print("=" * 60)
    print("PASO 3: Guardando texto procesado")
    print("=" * 60)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        path = save_processed_text(doc, PROCESSED_DIR)
        print(f"  Guardado: {path.name}")
    print()

    # Paso 4: Dividir en chunks
    print("=" * 60)
    print("PASO 4: Dividiendo en chunks")
    print("=" * 60)
    chunks = chunk_all_documents(documents)
    print()

    # Paso 5: Crear embeddings y vector store
    print("=" * 60)
    print("PASO 5: Creando embeddings y vector store")
    print("=" * 60)
    embedding_model = get_embedding_model(embedding_model_key)
    vector_store = create_vector_store(chunks, embedding_model)

    # Resumen
    pdf_count = len([d for d in documents if d.get("source_type") != "web"])
    web_count = len([d for d in documents if d.get("source_type") == "web"])
    print("\n" + "=" * 60)
    print("INGESTA COMPLETADA")
    print("=" * 60)
    print(f"  PDFs procesados: {pdf_count}")
    print(f"  Paginas web procesadas: {web_count}")
    print(f"  Total documentos: {len(documents)}")
    print(f"  Chunks creados: {len(chunks)}")
    print(f"  Modelo de embedding: {embedding_model_key}")
    print(f"  Vector store listo para consultas")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de ingesta: Normatividad Uninorte -> ChromaDB"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Directorio con PDFs (default: descarga automatica)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        choices=["minilm-multilingual", "mpnet-multilingual"],
        help="Modelo de embedding a usar",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Forzar re-descarga completa desde la web de Uninorte",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None
    run_ingestion(
        pdf_dir=pdf_dir,
        embedding_model_key=args.embedding_model,
        force_download=args.download,
    )
