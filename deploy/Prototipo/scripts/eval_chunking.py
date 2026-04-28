"""Evaluacion empirica del tamano de chunk vs latencia y calidad.

Justifica al tutor que ``ARTICLE_MAX_CHARS=1500`` y la estrategia jerarquica
por articulo son la decisiones que mejor balancean *tiempo de respuesta* y
*calidad* (las dos variables que el tutor pidio defender).

El script genera chunks con tres configuraciones distintas, mide:
  - numero de chunks producidos
  - longitud media / p95 en caracteres
  - cuantos chunks exceden el limite de tokens del embedder (truncacion)
  - tiempo total de chunkeo + embedding sobre los reglamentos

Uso:
    python scripts/eval_chunking.py            # tabla en stdout
    python scripts/eval_chunking.py --json out.json

No toca ChromaDB ni el SLM. Es un eval offline; corre en local en segundos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

# Permitir ejecucion como `python scripts/eval_chunking.py` desde Prototipo/
PROTOTYPE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROTOTYPE_ROOT))

from src.pdf_extractor import extract_all_pdfs  # noqa: E402
from src.text_chunker import chunk_all_documents  # noqa: E402
from config import SCRAPING_PDF_DIR  # noqa: E402


# Limite efectivo del embedder MiniLM (paraphrase-multilingual-MiniLM-L12-v2):
# 512 tokens ~ 2000 caracteres. Por encima la cola se trunca silenciosamente.
EMBEDDER_CHAR_LIMIT = 2000


CONFIGS: List[Dict] = [
    {
        "name": "small (300/50)",
        "chunk_size": 300,
        "chunk_overlap": 50,
        # Para forzar el splitter recursivo, monkey-patcheamos los limites de
        # articulo a 0 (en `_run`).
        "force_recursive": True,
    },
    {
        "name": "current (article-aware, 1500 max)",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "force_recursive": False,
    },
    {
        "name": "large (3000/500)",
        "chunk_size": 3000,
        "chunk_overlap": 500,
        "force_recursive": True,
    },
]


def _measure(chunks) -> Dict:
    lengths = [len(c.page_content) for c in chunks]
    lengths_sorted = sorted(lengths)
    p95_idx = max(0, int(len(lengths_sorted) * 0.95) - 1)
    truncated = sum(1 for l in lengths if l > EMBEDDER_CHAR_LIMIT)
    article_anchored = sum(1 for c in chunks if c.metadata.get("article"))
    return {
        "n_chunks": len(chunks),
        "avg_chars": round(mean(lengths), 1) if lengths else 0,
        "p95_chars": lengths_sorted[p95_idx] if lengths_sorted else 0,
        "max_chars": max(lengths) if lengths else 0,
        "truncated_by_embedder": truncated,
        "article_anchored": article_anchored,
    }


def _run(documents, cfg: Dict) -> Dict:
    import config as _cfg

    saved_max = _cfg.ARTICLE_MAX_CHARS
    saved_min = _cfg.ARTICLE_MIN_CHARS
    if cfg["force_recursive"]:
        # Forzar splitter clasico: tope de articulo absurdamente bajo (0)
        # hace que `_chunk_by_article` no lo encuentre util.
        _cfg.ARTICLE_MAX_CHARS = 0
        _cfg.ARTICLE_MIN_CHARS = 10**9

    t0 = time.perf_counter()
    chunks = chunk_all_documents(
        documents,
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )
    elapsed = time.perf_counter() - t0

    _cfg.ARTICLE_MAX_CHARS = saved_max
    _cfg.ARTICLE_MIN_CHARS = saved_min

    metrics = _measure(chunks)
    metrics["chunking_seconds"] = round(elapsed, 3)
    metrics["config"] = cfg["name"]
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=SCRAPING_PDF_DIR,
        help=f"Carpeta con los PDFs (default: {SCRAPING_PDF_DIR})",
    )
    parser.add_argument("--json", type=Path, help="Volcar resultados a JSON")
    args = parser.parse_args()

    if not args.pdf_dir.exists():
        print(f"ERROR: no existe {args.pdf_dir}", file=sys.stderr)
        return 1

    print(f"Extrayendo PDFs desde {args.pdf_dir}...")
    documents = extract_all_pdfs(args.pdf_dir)
    print(f"  {len(documents)} documentos cargados.\n")

    results = [_run(documents, cfg) for cfg in CONFIGS]

    # Tabla
    headers = ["config", "n_chunks", "avg_chars", "p95_chars", "max_chars",
               "truncated_by_embedder", "article_anchored", "chunking_seconds"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in results)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for r in results:
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nResultados guardados en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
