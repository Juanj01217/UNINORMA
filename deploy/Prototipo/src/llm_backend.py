"""Adaptador del responder SLM.

Selecciona el motor de inferencia segun ``LLM_BACKEND``:

- ``ollama``  -> ``langchain_community.llms.Ollama`` (default, x86 y ARM CPU).
- ``rkllm``   -> NPU del Rockchip RK3588/RK3588S (Orange Pi 5 / 5 Pro / 5 Plus)
                via ``rkllm-toolkit`` / ``librkllmrt.so``.

El adaptador RKLLM expone la misma interfaz minima que usa el resto del
pipeline (``invoke(prompt: str) -> str``), de modo que ``rag_chain.create_llm``
puede sustituir uno por otro sin tocar el grafo de LangChain.

Si el binario / binding RKLLM no esta disponible (entorno x86 o instalacion
incompleta), se levanta ``RKLLMUnavailableError`` y el caller decide si
caer a Ollama o fallar duro. Esto evita que el contenedor x86 falle al
importar este modulo.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

_logger = logging.getLogger(__name__)


class RKLLMUnavailableError(RuntimeError):
    """Se lanza cuando el runtime RKLLM no se puede inicializar."""


class RKLLMRunner:
    """Wrapper minimo sobre el binding Python de ``rkllm-runtime``.

    El binding se importa de forma perezosa para que la imagen x86 no falle
    al cargar este modulo. La instancia es *thread-safe a nivel de proceso*
    pero NO concurrente: la NPU del RK3588 sirve una secuencia a la vez.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 300,
        temperature: float = 0.1,
        target_platform: str = "rk3588",
    ) -> None:
        if not os.path.exists(model_path):
            raise RKLLMUnavailableError(
                f"No se encontro el modelo .rkllm en {model_path}. "
                "Convierte el modelo con rkllm-toolkit o monta el volumen "
                "./deploy/models con el .rkllm pre-generado."
            )

        try:
            # rkllm-runtime publica el binding como `rkllm` en PyPI ARM64
            # (ver https://github.com/airockchip/rknn-llm). En x86 este
            # import falla y caemos al else.
            from rkllm.api import RKLLM  # type: ignore
        except ImportError as exc:
            raise RKLLMUnavailableError(
                "rkllm-runtime no esta instalado. Solo disponible en ARM64 "
                "con NPU RK3588. Instala con `pip install rkllm-runtime` "
                "dentro del contenedor ARM64."
            ) from exc

        _logger.info("Cargando modelo RKLLM: %s (NPU %s)", model_path, target_platform)
        self._llm = RKLLM(model_path=model_path, target_platform=target_platform)
        self._llm.load()
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        _logger.info("RKLLM listo.")

    def invoke(self, prompt: str) -> str:
        """Ejecuta inferencia sincrona y devuelve el texto generado."""
        return self._llm.run(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )

    def stream(self, prompt: str):
        """Streaming token-a-token si el binding lo expone; si no, single chunk.

        Mantiene la firma usada por la cadena RAG (``llm.stream(prompt)`` ->
        iterable de tokens/strings).
        """
        run_stream = getattr(self._llm, "run_stream", None)
        if callable(run_stream):
            yield from run_stream(
                prompt,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
            )
        else:
            yield self.invoke(prompt)

    def __call__(self, prompt: str) -> str:  # compat con LangChain Runnable simple
        return self.invoke(prompt)


def create_responder_llm(
    backend: str,
    *,
    ollama_factory,
    rkllm_model_path: str,
    rkllm_target_platform: str,
    max_tokens: int,
    temperature: float,
) -> object:
    """Factory unica para el LLM responder.

    ``ollama_factory`` es una callable que devuelve la instancia ``Ollama`` ya
    parametrizada (la pasa el caller para no duplicar parametros). Se usa
    siempre que el backend sea ``ollama`` o cuando el runtime RKLLM no este
    disponible (fallback transparente).
    """
    backend = (backend or "ollama").lower()

    if backend == "rkllm":
        try:
            return RKLLMRunner(
                model_path=rkllm_model_path,
                max_new_tokens=max_tokens,
                temperature=temperature,
                target_platform=rkllm_target_platform,
            )
        except RKLLMUnavailableError as exc:
            _logger.warning(
                "LLM_BACKEND=rkllm pero el runtime no esta disponible (%s). "
                "Cayendo a Ollama.", exc,
            )
            return ollama_factory()

    if backend != "ollama":
        _logger.warning("LLM_BACKEND=%s desconocido; usando ollama.", backend)
    return ollama_factory()
