"""Gestion de modelos de embedding multilingue.

Dos backends disponibles, controlados por ``config.EMBEDDER_BACKEND``:

- ``st`` (default): ``HuggingFaceEmbeddings`` sobre PyTorch + sentence-transformers.
  Es la ruta probada; corre bien en x86 CPU/GPU y en ARM CPU. Trae torch como
  dependencia (~700 MB extra en la imagen).

- ``onnx``: el mismo MiniLM exportado a ONNX int8, servido con ``onnxruntime``.
  Pensado para Orange Pi 5 Pro (RK3588): ~2x menos RAM, ~1.5x mas rapido en
  ARM64, y elimina torch en runtime. Requiere que el modelo este pre-exportado
  (ver scripts/export_embedder_onnx.py).

Ambos backends devuelven una clase compatible con ``Embeddings`` de LangChain
(``embed_query`` / ``embed_documents``), por lo que ChromaDB y el retriever no
necesitan cambios.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL, EMBEDDER_BACKEND

_logger = logging.getLogger(__name__)


class _ONNXMiniLMEmbeddings:
    """Embedder LangChain-compatible sobre onnxruntime + tokenizers.

    Carga el modelo cuantizado int8 desde HuggingFace o desde un path local
    (``model_dir``). Aplica mean pooling + L2 norm para igualar el comportamiento
    de ``sentence-transformers`` (mismo espacio vectorial -> ChromaDB existente
    sigue siendo valida si el modelo base es el mismo).
    """

    def __init__(self, model_id: str, model_dir: str | None = None) -> None:
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "EMBEDDER_BACKEND=onnx requiere `optimum[onnxruntime]` y "
                "`transformers`. Instala con: pip install optimum[onnxruntime]"
            ) from exc

        source = model_dir or model_id
        _logger.info("Cargando embedder ONNX: %s", source)
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        # `file_name="model_quantized.onnx"` carga la version int8 si existe;
        # si no, optimum cae a la fp32 sin romper.
        self._model = ORTModelForFeatureExtraction.from_pretrained(
            source, file_name="model_quantized.onnx", export=model_dir is None,
        )
        _logger.info("Embedder ONNX cargado.")

    def _encode(self, texts: List[str]) -> List[List[float]]:
        import numpy as np

        enc = self._tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        outputs = self._model(**enc)
        # last_hidden_state: (batch, seq, hidden). Mean pooling enmascarado.
        last_hidden = outputs.last_hidden_state
        mask = enc["attention_mask"][..., None].astype(last_hidden.dtype)
        summed = (last_hidden * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        pooled = summed / counts
        # L2 normalize para coincidir con normalize_embeddings=True del backend ST.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
        return (pooled / norms).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


def get_embedding_model(model_key: str = DEFAULT_EMBEDDING_MODEL):
    """Carga el embedder segun ``EMBEDDER_BACKEND``.

    Args:
        model_key: Clave del diccionario ``EMBEDDING_MODELS`` en config.

    Returns:
        Embedder compatible con LangChain (``embed_query`` / ``embed_documents``).
    """
    model_name = EMBEDDING_MODELS[model_key]

    if EMBEDDER_BACKEND == "onnx":
        # En ONNX el "model_id" puede ser el repo HF o un dir local exportado.
        return _ONNXMiniLMEmbeddings(model_id=f"sentence-transformers/{model_name}")

    if EMBEDDER_BACKEND not in {"st", "sentence-transformers"}:
        _logger.warning(
            "EMBEDDER_BACKEND=%s desconocido; usando sentence-transformers.",
            EMBEDDER_BACKEND,
        )

    print(f"Cargando modelo de embedding: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Modelo de embedding cargado exitosamente")
    return embeddings
