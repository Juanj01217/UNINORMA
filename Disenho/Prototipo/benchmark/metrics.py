"""Metricas personalizadas para evaluacion del sistema RAG."""
import time
import re
from dataclasses import dataclass, field, asdict
from typing import List

import psutil
import numpy as np


@dataclass
class BenchmarkResult:
    """Resultado de una consulta de benchmark."""
    question_id: str = ""
    model_name: str = ""
    question: str = ""
    answer: str = ""
    expected_source: str = ""
    retrieved_sources: List[str] = field(default_factory=list)
    category: str = ""
    difficulty: str = ""
    latency_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    retrieval_hit: bool = False
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    hallucination_detected: bool = False
    no_answer_correct: bool = False

    def to_dict(self):
        return asdict(self)


def measure_latency(func, *args, **kwargs):
    """Mide el tiempo de ejecucion de una funcion."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def get_memory_usage_mb() -> float:
    """Retorna el uso de memoria del proceso actual en MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def check_retrieval_hit(
    retrieved_sources: List[str],
    expected_source: str,
) -> bool:
    """Verifica si la fuente esperada esta entre los documentos recuperados."""
    if expected_source == "NONE":
        return True  # Para preguntas de negacion no hay fuente esperada

    for source in retrieved_sources:
        if expected_source in source or source in expected_source:
            return True
    return False


def compute_answer_relevancy(
    question: str,
    answer: str,
    embedding_model,
) -> float:
    """
    Calcula similitud coseno entre pregunta y respuesta como proxy de relevancia.
    """
    try:
        q_emb = embedding_model.encode([question])[0]
        a_emb = embedding_model.encode([answer])[0]

        cosine_sim = np.dot(q_emb, a_emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-8
        )
        return float(max(0, min(1, cosine_sim)))
    except Exception:
        return 0.0


def compute_faithfulness(answer: str, context: str) -> float:
    """
    Estima fidelidad verificando que fracciones de la respuesta
    aparecen en el contexto (overlap de oraciones).
    """
    if not answer.strip() or not context.strip():
        return 0.0

    # Dividir respuesta en oraciones
    answer_sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    if not answer_sentences:
        return 0.0

    context_lower = context.lower()
    grounded_count = 0

    for sentence in answer_sentences:
        # Verificar si palabras clave de la oracion estan en el contexto
        words = [w for w in sentence.lower().split() if len(w) > 3]
        if not words:
            grounded_count += 1
            continue

        matches = sum(1 for w in words if w in context_lower)
        if matches / len(words) >= 0.5:
            grounded_count += 1

    return grounded_count / len(answer_sentences)


def detect_hallucination(answer: str, context: str) -> bool:
    """
    Deteccion simple de alucinaciones: verifica si la respuesta
    introduce numeros, fechas o entidades no presentes en el contexto.
    """
    if not answer.strip() or not context.strip():
        return False

    # Extraer numeros de la respuesta y el contexto
    answer_numbers = set(re.findall(r'\b\d+\b', answer))
    context_numbers = set(re.findall(r'\b\d+\b', context))

    # Numeros en la respuesta que no estan en el contexto
    new_numbers = answer_numbers - context_numbers
    # Filtrar numeros muy comunes (1, 2, etc.)
    significant_new = {n for n in new_numbers if int(n) > 10}

    if len(significant_new) > 2:
        return True

    # Verificar si la respuesta indica "no encontre informacion"
    no_info_patterns = [
        "no encontre",
        "no tengo informacion",
        "no se encuentra",
        "no dispongo",
        "no hay informacion",
    ]
    answer_lower = answer.lower()
    if any(p in answer_lower for p in no_info_patterns):
        return False  # Respuesta correcta de negacion

    return False


def check_no_answer_correct(
    answer: str,
    expected_source: str,
) -> bool:
    """
    Para preguntas donde la respuesta NO esta en los documentos,
    verifica si el modelo correctamente indica que no tiene la informacion.
    """
    if expected_source != "NONE":
        return True  # No aplica para preguntas normales

    no_info_patterns = [
        "no encontre",
        "no tengo informacion",
        "no se encuentra",
        "no dispongo",
        "no hay informacion",
        "no puedo responder",
        "no esta disponible",
        "no aparece en",
        "no menciona",
    ]
    answer_lower = answer.lower()
    return any(p in answer_lower for p in no_info_patterns)
