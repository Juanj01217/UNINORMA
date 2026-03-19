"""
Scraper completo para la normatividad de Uninorte.

Maneja 3 tipos de contenido:
  1. PDFs directos (descarga)
  2. Sub-paginas que contienen PDFs (sigue el enlace y descarga)
  3. Paginas web sin PDF (extrae el texto HTML)
"""
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


NORMATIVIDAD_URL = "https://www.uninorte.edu.co/web/sobre-nosotros/normatividad"
BASE_URL = "https://www.uninorte.edu.co"

# Sub-paginas conocidas que contienen PDFs adicionales
SUB_PAGES_WITH_PDFS = {
    "/reglamento_estudiantil",
    "/web/sobre-nosotros/politica-no-violencia",
    "/web/sostenibilidad/informe-sostenibilidad",
    "/regimen_tributario_especial_fun",
}

# Paginas que son contenido web puro (texto HTML)
WEB_CONTENT_PAGES = {
    "/politica-de-privacidad-de-datos",
    "/web/guest/politica-de-privacidad-de-datos",
}

# Paginas con contenido embebido que no se puede scrapear facilmente
SKIP_PAGES = {
    "/web/universidadincluyente/ruta-de-atencion-para-la-diversidad-estudiantil",
    "/web/comunicaciones/portal-creativo",
}

# Mapeo de URLs a titulos legibles (para enlaces con texto generico)
URL_TITLE_MAP = {
    "/reglamento_estudiantil": "Reglamento Estudiantil",
    "/web/sobre-nosotros/politica-no-violencia": "Politica contra Violencia y Acoso",
    "/web/sostenibilidad/informe-sostenibilidad": "Informe de Sostenibilidad",
    "/regimen_tributario_especial_fun": "Regimen Tributario Especial",
    "/politica-de-privacidad-de-datos": "Politica de Privacidad de Datos",
    "/web/guest/politica-de-privacidad-de-datos": "Politica de Privacidad de Datos",
}


def _create_session() -> requests.Session:
    """Crea sesion HTTP con reintentos y headers."""
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    })
    return session


def _classify_link(href: str) -> str:
    """Clasifica un enlace: 'pdf', 'subpage', 'webcontent', o 'skip'."""
    if href.endswith(".pdf"):
        return "pdf"

    # Extraer path relativo
    path = href
    if href.startswith("http"):
        from urllib.parse import urlparse
        path = urlparse(href).path

    if path in SKIP_PAGES:
        return "skip"
    if path in WEB_CONTENT_PAGES:
        return "webcontent"
    if path in SUB_PAGES_WITH_PDFS:
        return "subpage"

    # Heuristica: si parece una pagina uninorte interna, tratarla como webcontent
    if "/web/" in path or path.startswith("/politica"):
        return "webcontent"

    return "skip"


def _normalize_url(href: str) -> str:
    """Normaliza un href relativo a URL completa."""
    if href.startswith("http"):
        return href
    return urljoin(BASE_URL, href)


def _download_pdf(session: requests.Session, url: str, dest_dir: Path) -> Optional[Path]:
    """Descarga un PDF y lo guarda en dest_dir. Retorna el path o None."""
    try:
        # Extraer nombre del archivo normalizado
        filename = unquote(url.split("/")[-1])
        # Limpiar caracteres problematicos y normalizar espacios
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        filename = filename.replace("%20", " ")

        filepath = dest_dir / filename

        # No re-descargar si ya existe
        if filepath.exists() and filepath.stat().st_size > 0:
            print(f"    (ya existe: {filename})")
            return filepath

        resp = session.get(url, timeout=30)
        resp.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(resp.content)

        return filepath
    except Exception as e:
        print(f"    ERROR descargando PDF {url}: {e}")
        return None


def _extract_text_from_html(soup: BeautifulSoup, url: str) -> str:
    """Extrae texto limpio de una pagina HTML sin duplicar contenido."""
    # Intentar encontrar el contenido principal
    content = None

    # Buscar en orden de prioridad los contenedores tipicos de Liferay/Uninorte
    for selector in [
        "div.journal-content-article",
        "div.c_cr",
        "div.web-content-display",
        "article",
        "main",
        "div.portlet-body",
    ]:
        content = soup.select_one(selector)
        if content and len(content.get_text(strip=True)) > 100:
            break

    if not content:
        # Fallback: body sin header/footer/nav
        content = soup.find("body")
        if content:
            for tag in content.find_all(["header", "footer", "nav", "script", "style"]):
                tag.decompose()

    if not content:
        return ""

    # Eliminar scripts, estilos, y elementos de navegacion
    for tag in content.find_all(["script", "style", "nav", "iframe"]):
        tag.decompose()

    # Extraer texto usando solo elementos de bloque (evita duplicacion)
    lines = []
    processed_elements = set()

    for element in content.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div", "td", "th", "blockquote"]
    ):
        # Evitar procesar elementos ya contenidos dentro de otros procesados
        if id(element) in processed_elements:
            continue

        # No procesar divs que contengan sub-elementos de bloque
        if element.name == "div":
            has_block_children = element.find(
                ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div", "table"]
            )
            if has_block_children:
                continue

        text = element.get_text(strip=True)
        if not text or len(text) < 3:
            continue

        # Marcar elementos hijos como ya procesados
        for child in element.find_all(True):
            processed_elements.add(id(child))

        if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            lines.append(f"\n\n## {text}\n")
        elif element.name == "li":
            lines.append(f"  - {text}")
        else:
            lines.append(f"\n{text}\n")

    text = "\n".join(lines)
    # Limpiar exceso de lineas vacias
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_document_from_html(
    title: str, text: str, url: str
) -> Dict[str, Any]:
    """Crea un documento compatible con el formato de pdf_extractor."""
    # Dividir texto en "paginas" virtuales de ~3000 chars para metadata consistente
    page_size = 3000
    pages = []
    for i in range(0, len(text), page_size):
        chunk = text[i:i + page_size]
        pages.append({
            "page_number": len(pages) + 1,
            "text": chunk.strip(),
        })

    if not pages:
        pages = [{"page_number": 1, "text": text}]

    # Generar un nombre de archivo seguro
    safe_name = re.sub(r"[^\w\-]", "_", title)[:80] + ".html"

    return {
        "filename": safe_name,
        "title": title,
        "num_pages": len(pages),
        "full_text": text,
        "pages": pages,
        "source_url": url,
        "source_type": "web",
    }


def _extract_title(link_tag, href: str) -> str:
    """Extrae un titulo significativo del enlace o su contexto."""
    from urllib.parse import urlparse

    # 1. Revisar mapeo conocido de URLs
    path = urlparse(href).path if href.startswith("http") else href
    if path in URL_TITLE_MAP:
        return URL_TITLE_MAP[path]

    # 2. Texto del enlace (si no es generico)
    text = link_tag.get_text(strip=True)
    generic = {"Ver más", "ver más", "Ver mas", "Descargar", "Aquí", "aquí", ""}
    if text and text not in generic:
        return text

    # 3. Buscar titulo en elemento padre o hermano anterior
    parent = link_tag.parent
    if parent:
        # Buscar headings o textos descriptivos en hermanos previos
        for sibling in parent.previous_siblings:
            if isinstance(sibling, Tag):
                sib_text = sibling.get_text(strip=True)
                if sib_text and len(sib_text) > 3 and sib_text not in generic:
                    return sib_text[:100]
            elif isinstance(sibling, str) and sibling.strip():
                return sibling.strip()[:100]
        # Buscar en el texto directo del padre
        for child in parent.children:
            if isinstance(child, str) and child.strip() and child.strip() not in generic:
                return child.strip()[:100]

    # 4. Extraer del nombre del archivo en la URL
    filename = href.split("/")[-1]
    if filename.endswith(".pdf"):
        from urllib.parse import unquote
        name = unquote(filename).replace(".pdf", "").replace("_", " ").replace("%20", " ")
        return re.sub(r"\s+", " ", name).strip()

    # 5. Fallback
    return path.strip("/").split("/")[-1].replace("-", " ").replace("_", " ").title() or "Sin titulo"


def scrape_normatividad(
    pdf_dest_dir: Path,
    session: requests.Session = None,
) -> Dict[str, Any]:
    """
    Scraping completo de la pagina de normatividad.

    Returns:
        Dict con:
          - pdfs_downloaded: int
          - web_documents: List[Dict] (documentos extraidos de paginas web)
          - errors: List[str]
    """
    if session is None:
        session = _create_session()

    pdf_dest_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "pdfs_downloaded": 0,
        "web_documents": [],
        "errors": [],
    }

    print(f"  Descargando pagina principal: {NORMATIVIDAD_URL}")
    try:
        page = session.get(NORMATIVIDAD_URL, timeout=15)
        page.raise_for_status()
    except Exception as e:
        results["errors"].append(f"No se pudo acceder a la pagina principal: {e}")
        return results

    soup = BeautifulSoup(page.text, "html.parser")

    # Buscar el contenedor principal de enlaces
    main_content = soup.find("div", class_="c_cr")
    if not main_content:
        # Intentar otro selector
        main_content = soup.find("div", class_="journal-content-article")
    if not main_content:
        results["errors"].append("No se encontro el contenedor principal de enlaces")
        return results

    # Recolectar todos los enlaces unicos (normalizar URLs para evitar duplicados)
    seen_urls = set()
    links_to_process = []

    for link in main_content.find_all("a", href=True):
        href = link["href"].strip()
        url = _normalize_url(href)

        # Normalizar: decodificar URL para detectar duplicados como
        # "Reg_profesor_%20junio_2021.pdf" y "Reg_profesor_ junio_2021.pdf"
        normalized_url = unquote(url).replace(" ", "_").lower()
        if url in seen_urls or normalized_url in seen_urls:
            continue
        seen_urls.add(url)
        seen_urls.add(normalized_url)

        title = _extract_title(link, href)
        link_type = _classify_link(href)

        links_to_process.append({
            "title": title,
            "url": url,
            "href": href,
            "type": link_type,
        })

    # Tambien buscar enlaces en el footer/pie de pagina legal
    footer_content = soup.find("footer") or soup.find("div", class_="footer")
    if footer_content:
        for link in footer_content.find_all("a", href=True):
            href = link["href"].strip()
            if ".pdf" in href:
                url = _normalize_url(href)
                if url not in seen_urls:
                    seen_urls.add(url)
                    title = link.get_text(strip=True) or "Sin titulo"
                    links_to_process.append({
                        "title": title,
                        "url": url,
                        "href": href,
                        "type": "pdf",
                    })

    print(f"  Encontrados {len(links_to_process)} enlaces en la pagina principal")

    # Procesar cada enlace segun su tipo
    for item in links_to_process:
        title = item["title"]
        url = item["url"]
        link_type = item["type"]

        if link_type == "pdf":
            print(f"  [PDF] {title}")
            path = _download_pdf(session, url, pdf_dest_dir)
            if path:
                print(f"    -> Descargado: {path.name}")
                results["pdfs_downloaded"] += 1
            time.sleep(0.5)

        elif link_type == "subpage":
            print(f"  [SUB-PAGINA] {title} -> {url}")
            _process_subpage(session, url, title, pdf_dest_dir, results)
            time.sleep(1)

        elif link_type == "webcontent":
            print(f"  [WEB] {title} -> {url}")
            _process_web_content(session, url, title, results)
            time.sleep(1)

        elif link_type == "skip":
            print(f"  [SKIP] {title} (contenido embebido/interactivo)")

    return results


def _process_subpage(
    session: requests.Session,
    url: str,
    parent_title: str,
    pdf_dest_dir: Path,
    results: Dict[str, Any],
) -> None:
    """Procesa una sub-pagina: descarga sus PDFs y extrae texto."""
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        results["errors"].append(f"Error accediendo sub-pagina {url}: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # Mejorar titulo si es generico
    page_title = _get_page_title(soup, url, parent_title)

    # Buscar PDFs en la sub-pagina
    pdf_count = 0
    seen_pdfs = set()
    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        if href.endswith(".pdf"):
            pdf_url = _normalize_url(href)
            if pdf_url in seen_pdfs:
                continue
            seen_pdfs.add(pdf_url)

            path = _download_pdf(session, pdf_url, pdf_dest_dir)
            if path:
                print(f"    -> PDF: {path.name}")
                results["pdfs_downloaded"] += 1
                pdf_count += 1
            time.sleep(0.5)

    # Tambien extraer texto de la propia pagina (puede tener info util)
    page_text = _extract_text_from_html(soup, url)
    if page_text and len(page_text) > 200:
        doc = _make_document_from_html(page_title, page_text, url)
        results["web_documents"].append(doc)
        print(f"    -> Texto web ({page_title}): {len(page_text)} caracteres")

    if pdf_count == 0 and not page_text:
        print(f"    -> Sin contenido extraible")


def _get_page_title(soup: BeautifulSoup, url: str, fallback: str) -> str:
    """Extrae el mejor titulo posible de una pagina web."""
    from urllib.parse import urlparse

    # 1. Revisar mapeo conocido
    path = urlparse(url).path
    if path in URL_TITLE_MAP:
        return URL_TITLE_MAP[path]

    # 2. Buscar <title> de la pagina
    title_tag = soup.find("title")
    if title_tag:
        page_title = title_tag.get_text(strip=True)
        # Limpiar sufijos comunes
        for suffix in [" - Universidad del Norte", " | Uninorte", " – Uninorte"]:
            page_title = page_title.replace(suffix, "")
        if page_title and len(page_title) > 3:
            return page_title.strip()

    # 3. Buscar <h1>
    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(strip=True)
        if h1_text and len(h1_text) > 3:
            return h1_text[:100]

    # 4. Fallback
    return fallback


def _process_web_content(
    session: requests.Session,
    url: str,
    title: str,
    results: Dict[str, Any],
) -> None:
    """Procesa una pagina web pura: extrae texto HTML."""
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        results["errors"].append(f"Error accediendo pagina web {url}: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # Mejorar titulo si es generico
    page_title = _get_page_title(soup, url, title)
    text = _extract_text_from_html(soup, url)

    if text and len(text) > 100:
        doc = _make_document_from_html(page_title, text, url)
        results["web_documents"].append(doc)
        print(f"    -> Extraido ({page_title}): {len(text)} caracteres, {doc['num_pages']} secciones")
    else:
        print(f"    -> Contenido insuficiente ({len(text)} chars)")
        results["errors"].append(f"Poco texto extraible de {url}")
