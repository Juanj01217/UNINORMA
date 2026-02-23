"""
Pipeline de ingesta: PDF -> texto -> chunks -> embeddings -> ChromaDB.

Uso:
    python ingest.py
    python ingest.py --pdf-dir ruta/a/pdfs
    python ingest.py --embedding-model mpnet-multilingual
    python ingest.py --download   (descarga PDFs automaticamente)
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import SCRAPING_PDF_DIR, RAW_PDF_DIR, PROCESSED_DIR, DEFAULT_EMBEDDING_MODEL
from src.pdf_extractor import extract_all_pdfs, save_processed_text
from src.text_chunker import chunk_all_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store


UNINORTE_NORMATIVIDAD_URL = "https://www.uninorte.edu.co/web/sobre-nosotros/normatividad"


def download_pdfs(dest_dir: Path) -> int:
    """Descarga PDFs de normatividad directamente desde la web de Uninorte."""
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Descargando desde: {UNINORTE_NORMATIVIDAD_URL}")

    # Configurar sesion con reintentos
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # Obtener la pagina
    page = session.get(UNINORTE_NORMATIVIDAD_URL, headers=headers, timeout=15)
    page.raise_for_status()
    soup = BeautifulSoup(page.text, "html.parser")

    # Buscar el div con los enlaces a PDFs
    div_reglamentos = soup.find("div", class_="c_cr")
    if not div_reglamentos:
        raise RuntimeError(
            "No se encontro el div con clase 'c_cr' en la pagina. "
            "La estructura de la web puede haber cambiado."
        )

    base_url = UNINORTE_NORMATIVIDAD_URL.split("/web")[0]
    count = 0

    for link in div_reglamentos.find_all("a", href=True):
        href = link["href"]
        if not href.endswith(".pdf"):
            continue

        if not href.startswith("http"):
            href = base_url + href

        try:
            pdf_response = session.get(href, headers=headers, timeout=15)
            pdf_response.raise_for_status()

            filename = href.split("/")[-1]
            filepath = dest_dir / filename

            with open(filepath, "wb") as f:
                f.write(pdf_response.content)

            print(f"    Descargado: {filename}")
            count += 1
            time.sleep(1)

        except Exception as e:
            print(f"    Error descargando {href}: {e}")

    return count


def copy_pdfs_to_data(source_dir: Path, dest_dir: Path) -> int:
    """Copia PDFs del directorio de scraping al directorio de datos del prototipo."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(source_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No se encontraron PDFs en {source_dir}.\n"
            f"Opciones:\n"
            f"  1. Ejecuta: python ingest.py --download\n"
            f"  2. Ejecuta el notebook WebScraping/dataCollection.ipynb\n"
            f"  3. Coloca PDFs manualmente en: {dest_dir}"
        )

    for pdf in pdf_files:
        dest = dest_dir / pdf.name
        if not dest.exists():
            shutil.copy2(pdf, dest)
            print(f"  Copiado: {pdf.name}")
        else:
            print(f"  Ya existe: {pdf.name}")

    return len(pdf_files)


def obtain_pdfs(dest_dir: Path, force_download: bool = False) -> int:
    """Obtiene los PDFs: busca en varias ubicaciones o descarga."""
    # Si ya hay PDFs en data/raw, usarlos
    existing = list(dest_dir.glob("*.pdf"))
    if existing and not force_download:
        print(f"  {len(existing)} PDFs ya presentes en data/raw")
        return len(existing)

    # Intentar copiar desde WebScraping/reglamentos
    if SCRAPING_PDF_DIR.exists() and list(SCRAPING_PDF_DIR.glob("*.pdf")):
        print(f"  Copiando desde {SCRAPING_PDF_DIR}")
        return copy_pdfs_to_data(SCRAPING_PDF_DIR, dest_dir)

    # Descargar directamente de la web
    print("  No se encontraron PDFs locales. Descargando desde Uninorte...")
    return download_pdfs(dest_dir)


def run_ingestion(
    pdf_dir: Path = None,
    embedding_model_key: str = DEFAULT_EMBEDDING_MODEL,
    force_download: bool = False,
) -> None:
    """Ejecuta el pipeline completo de ingesta."""

    # Paso 1: Obtener PDFs
    if pdf_dir is None:
        pdf_dir = RAW_PDF_DIR
        print("=" * 60)
        print("PASO 1: Obteniendo PDFs de normatividad")
        print("=" * 60)
        count = obtain_pdfs(RAW_PDF_DIR, force_download)
        print(f"-> {count} PDFs disponibles\n")
    else:
        print("=" * 60)
        print(f"PASO 1: Usando PDFs de {pdf_dir}")
        print("=" * 60 + "\n")

    # Verificar que hay PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("ERROR: No hay PDFs disponibles para procesar.")
        print("Ejecuta: python ingest.py --download")
        sys.exit(1)

    # Paso 2: Extraer texto
    print("=" * 60)
    print("PASO 2: Extrayendo texto de PDFs")
    print("=" * 60)
    documents = extract_all_pdfs(pdf_dir)
    print(f"-> {len(documents)} documentos procesados\n")

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
    print("\n" + "=" * 60)
    print("INGESTA COMPLETADA")
    print("=" * 60)
    print(f"  Documentos procesados: {len(documents)}")
    print(f"  Chunks creados: {len(chunks)}")
    print(f"  Modelo de embedding: {embedding_model_key}")
    print(f"  Vector store listo para consultas")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de ingesta: PDF -> ChromaDB"
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
        help="Forzar descarga de PDFs desde la web de Uninorte",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None
    run_ingestion(
        pdf_dir=pdf_dir,
        embedding_model_key=args.embedding_model,
        force_download=args.download,
    )
