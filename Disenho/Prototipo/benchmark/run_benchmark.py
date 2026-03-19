"""
Ejecutor principal de benchmarks: evalua multiples modelos con preguntas de prueba.

Uso:
    python -m benchmark.run_benchmark
    python -m benchmark.run_benchmark --models qwen2.5:3b phi3:mini
    python -m benchmark.run_benchmark --questions benchmark/test_questions.json
"""
import argparse
import json
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SLM_MODELS, DEFAULT_EMBEDDING_MODEL
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store, get_retriever
from src.rag_chain import create_rag_chain, query_rag
from src.ollama_client import check_ollama_running, get_available_models
from benchmark.metrics import (
    BenchmarkResult,
    measure_latency,
    get_memory_usage_mb,
    check_retrieval_hit,
    compute_answer_relevancy,
    compute_faithfulness,
    detect_hallucination,
    check_no_answer_correct,
)


def load_test_questions(path: Path) -> List[dict]:
    """Carga las preguntas de prueba desde JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def run_single_model_benchmark(
    model_name: str,
    questions: List[dict],
    retriever,
    raw_embedding_model,
) -> List[BenchmarkResult]:
    """Ejecuta todas las preguntas con un solo modelo y recolecta metricas."""
    print(f"\n{'='*60}")
    print(f"EVALUANDO: {model_name}")
    print(f"{'='*60}")

    chain = create_rag_chain(retriever, model_name)
    results = []

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        expected_source = q["expected_source"]

        print(f"  [{i+1}/{len(questions)}] {qid}: {question[:60]}...")

        mem_before = get_memory_usage_mb()

        try:
            rag_result, latency = measure_latency(
                query_rag, chain, question, model_name
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            result = BenchmarkResult(
                question_id=qid,
                model_name=model_name,
                question=question,
                answer=f"ERROR: {e}",
                expected_source=expected_source,
                category=q.get("category", ""),
                difficulty=q.get("difficulty", ""),
                latency_seconds=-1,
            )
            results.append(result)
            continue

        mem_after = get_memory_usage_mb()

        answer = rag_result["answer"]
        source_docs = rag_result.get("source_documents", [])
        retrieved_sources = [
            doc.metadata.get("source", "") for doc in source_docs
        ]

        # Concatenar contexto para metricas
        context = "\n".join(doc.page_content for doc in source_docs)

        # Calcular metricas
        retrieval_hit = check_retrieval_hit(retrieved_sources, expected_source)
        relevancy = compute_answer_relevancy(
            question, answer, raw_embedding_model
        )
        faithfulness_score = compute_faithfulness(answer, context)
        hallucination = detect_hallucination(answer, context)
        no_answer = check_no_answer_correct(answer, expected_source)

        result = BenchmarkResult(
            question_id=qid,
            model_name=model_name,
            question=question,
            answer=answer,
            expected_source=expected_source,
            retrieved_sources=retrieved_sources,
            category=q.get("category", ""),
            difficulty=q.get("difficulty", ""),
            latency_seconds=latency,
            memory_usage_mb=max(0, mem_after - mem_before),
            retrieval_hit=retrieval_hit,
            answer_relevancy=relevancy,
            faithfulness=faithfulness_score,
            hallucination_detected=hallucination,
            no_answer_correct=no_answer,
        )
        results.append(result)

        print(f"    Latencia: {latency:.2f}s | Relevancia: {relevancy:.2f} | "
              f"Fidelidad: {faithfulness_score:.2f} | Retrieval: {'OK' if retrieval_hit else 'MISS'}")

    return results


def generate_summary(
    all_results: List[BenchmarkResult],
) -> Dict[str, Dict]:
    """Agrega resultados por modelo en una tabla comparativa."""
    models = {}
    for r in all_results:
        if r.model_name not in models:
            models[r.model_name] = []
        models[r.model_name].append(r)

    summary = {}
    for model_name, results in models.items():
        valid = [r for r in results if r.latency_seconds >= 0]
        if not valid:
            continue

        summary[model_name] = {
            "total_questions": len(results),
            "successful_answers": len(valid),
            "avg_latency_seconds": sum(r.latency_seconds for r in valid) / len(valid),
            "max_latency_seconds": max(r.latency_seconds for r in valid),
            "min_latency_seconds": min(r.latency_seconds for r in valid),
            "avg_memory_mb": sum(r.memory_usage_mb for r in valid) / len(valid),
            "retrieval_accuracy": sum(1 for r in valid if r.retrieval_hit) / len(valid),
            "avg_answer_relevancy": sum(r.answer_relevancy for r in valid) / len(valid),
            "avg_faithfulness": sum(r.faithfulness for r in valid) / len(valid),
            "hallucination_rate": sum(1 for r in valid if r.hallucination_detected) / len(valid),
            "no_answer_accuracy": sum(
                1 for r in valid if r.expected_source == "NONE" and r.no_answer_correct
            ) / max(1, sum(1 for r in valid if r.expected_source == "NONE")),
        }

    return summary


def save_results(
    all_results: List[BenchmarkResult],
    summary: Dict,
    output_dir: Path,
) -> None:
    """Guarda resultados en JSON y CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resultados crudos en JSON
    raw_path = output_dir / f"{timestamp}_raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(
            [r.to_dict() for r in all_results],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Resultados crudos: {raw_path}")

    # Resumen en JSON
    summary_json_path = output_dir / f"{timestamp}_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Resumen JSON: {summary_json_path}")

    # Resumen en CSV
    summary_csv_path = output_dir / f"{timestamp}_summary.csv"
    if summary:
        fieldnames = ["model"] + list(next(iter(summary.values())).keys())
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for model_name, metrics in summary.items():
                row = {"model": model_name, **metrics}
                writer.writerow(row)
        print(f"Resumen CSV: {summary_csv_path}")


def print_summary_table(summary: Dict) -> None:
    """Imprime tabla resumen formateada en consola."""
    if not summary:
        print("Sin resultados para mostrar.")
        return

    print(f"\n{'='*90}")
    print("RESUMEN COMPARATIVO DE MODELOS")
    print(f"{'='*90}")

    header = (
        f"{'Modelo':<20} {'Latencia':>10} {'Retrieval':>10} "
        f"{'Relevancia':>10} {'Fidelidad':>10} {'Alucin.':>10}"
    )
    print(header)
    print("-" * 90)

    for model_name, m in summary.items():
        row = (
            f"{model_name:<20} "
            f"{m['avg_latency_seconds']:>8.2f}s "
            f"{m['retrieval_accuracy']:>9.1%} "
            f"{m['avg_answer_relevancy']:>10.3f} "
            f"{m['avg_faithfulness']:>10.3f} "
            f"{m['hallucination_rate']:>9.1%}"
        )
        print(row)

    print(f"{'='*90}")


def run_full_benchmark(
    models: List[str] = None,
    questions_path: Path = None,
    output_dir: Path = None,
) -> None:
    """Ejecuta benchmark completo con todos los modelos y guarda resultados."""
    if models is None:
        models = SLM_MODELS
    if questions_path is None:
        questions_path = Path(__file__).parent / "test_questions.json"
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    # Verificaciones
    if not check_ollama_running():
        print("ERROR: Ollama no esta activo. Inicialo antes de continuar.")
        sys.exit(1)

    available = get_available_models(models)
    runnable = [m for m in models if available.get(m, False)]

    if not runnable:
        print("ERROR: Ningun modelo solicitado esta instalado.")
        print("Modelos solicitados:", models)
        print("Instala al menos uno: ollama pull <modelo>")
        sys.exit(1)

    skipped = [m for m in models if m not in runnable]
    if skipped:
        print(f"ADVERTENCIA: Modelos no disponibles (se omiten): {skipped}")

    # Cargar preguntas
    questions = load_test_questions(questions_path)
    print(f"Preguntas de prueba: {len(questions)}")
    print(f"Modelos a evaluar: {runnable}")

    # Cargar componentes compartidos
    print("\nCargando embedding model y vector store...")
    embedding_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    vector_store = load_vector_store(embedding_model)
    retriever = get_retriever(vector_store)

    # Cargar modelo raw para metricas de relevancia
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODELS
    raw_model = SentenceTransformer(EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL])

    # Ejecutar benchmarks
    all_results = []
    for model_name in runnable:
        model_results = run_single_model_benchmark(
            model_name, questions, retriever, raw_model
        )
        all_results.extend(model_results)

    # Generar y guardar resumen
    summary = generate_summary(all_results)
    save_results(all_results, summary, output_dir)
    print_summary_table(summary)

    print(f"\nBenchmark completado. Resultados en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAG multi-modelo")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Modelos a evaluar (default: todos los configurados)",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Ruta al JSON de preguntas",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio para resultados",
    )
    args = parser.parse_args()

    run_full_benchmark(
        models=args.models,
        questions_path=Path(args.questions) if args.questions else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
