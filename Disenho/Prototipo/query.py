"""
CLI para probar consultas al sistema RAG.

Uso:
    python query.py "Tu pregunta aqui"
    python query.py --model qwen2.5:3b "Tu pregunta aqui"
    python query.py --interactive
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DEFAULT_SLM_MODEL, DEFAULT_EMBEDDING_MODEL
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store, get_retriever
from src.rag_chain import create_rag_chain, query_rag, format_response_with_sources
from src.ollama_client import check_ollama_running


def setup_chain(model_name: str = DEFAULT_SLM_MODEL):
    """Inicializa el sistema RAG completo."""
    if not check_ollama_running():
        print("ERROR: Ollama no esta activo. Inicialo antes de continuar.")
        sys.exit(1)

    print(f"Cargando sistema RAG con modelo: {model_name}")
    embedding_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    vector_store = load_vector_store(embedding_model)
    retriever = get_retriever(vector_store)
    chain = create_rag_chain(retriever, model_name)
    print("Sistema listo.\n")
    return chain


def single_query(question: str, model_name: str):
    """Ejecuta una consulta unica."""
    chain = setup_chain(model_name)
    print(f"Pregunta: {question}\n")
    print("Generando respuesta...\n")

    result = query_rag(chain, question, model_name)
    formatted = format_response_with_sources(result)
    print(formatted)


def interactive_mode(model_name: str):
    """Modo interactivo: multiples preguntas en sesion."""
    chain = setup_chain(model_name)

    print("=" * 60)
    print("MODO INTERACTIVO - Asistente de Normatividad Uninorte")
    print(f"Modelo: {model_name}")
    print("Escribe 'salir' para terminar")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Tu pregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSesion terminada.")
            break

        if not question:
            continue
        if question.lower() in ("salir", "exit", "quit"):
            print("Sesion terminada.")
            break

        print("\nGenerando respuesta...\n")
        result = query_rag(chain, question, model_name)
        formatted = format_response_with_sources(result)
        print(formatted)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI de consultas al RAG de normatividad Uninorte"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Pregunta a consultar",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_SLM_MODEL,
        help=f"Modelo SLM a usar (default: {DEFAULT_SLM_MODEL})",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Modo interactivo (multiples preguntas)",
    )
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.model)
    elif args.question:
        single_query(args.question, args.model)
    else:
        parser.print_help()
        print("\nEjemplos:")
        print('  python query.py "Cuales son los derechos de los egresados?"')
        print('  python query.py --model phi3:mini "Que dice el reglamento de profesores?"')
        print("  python query.py --interactive")
