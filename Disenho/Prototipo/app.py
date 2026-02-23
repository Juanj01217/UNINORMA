"""
Interfaz Gradio para el asistente de normatividad Uninorte.

Uso:
    python app.py
    python app.py --port 7860
    python app.py --share  (para URL publica temporal)
"""
import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))
from config import SLM_MODELS, DEFAULT_SLM_MODEL, DEFAULT_EMBEDDING_MODEL
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store, get_retriever
from src.rag_chain import create_rag_chain, query_rag, format_response_with_sources
from src.ollama_client import check_ollama_running, get_available_models

# Estado global
_current_chain = None
_current_model = None
_retriever = None


def _init_retriever():
    """Inicializa el retriever (una sola vez)."""
    global _retriever
    if _retriever is None:
        print("Cargando modelo de embedding y vector store...")
        embedding_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        vector_store = load_vector_store(embedding_model)
        _retriever = get_retriever(vector_store)
        print("Retriever listo.")
    return _retriever


def switch_model(model_name: str) -> str:
    """Cambia el modelo SLM activo."""
    global _current_chain, _current_model

    if not check_ollama_running():
        return "Ollama no esta activo. Inicialo antes de continuar."

    try:
        retriever = _init_retriever()
        _current_chain = create_rag_chain(retriever, model_name)
        _current_model = model_name
        return f"Modelo cambiado a: {model_name}"
    except Exception as e:
        return f"Error al cargar modelo {model_name}: {e}"


def get_status() -> str:
    """Obtiene el estado actual del sistema."""
    lines = []

    if check_ollama_running():
        lines.append("Ollama: Activo")
    else:
        lines.append("Ollama: INACTIVO - Inicia Ollama")
        return "\n".join(lines)

    available = get_available_models()
    installed = [m for m, v in available.items() if v]
    lines.append(f"Modelos instalados: {len(installed)}/{len(available)}")

    if _current_model:
        lines.append(f"Modelo activo: {_current_model}")
    else:
        lines.append("Modelo activo: Ninguno (selecciona uno)")

    return "\n".join(lines)


def chat_response(message: str, history: list) -> str:
    """Procesa un mensaje del usuario y genera respuesta."""
    global _current_chain, _current_model

    if not message.strip():
        return "Por favor, escribe una pregunta."

    if not check_ollama_running():
        return (
            "Ollama no esta activo. Por favor, inicia Ollama antes de hacer consultas.\n\n"
            "- **Windows**: Abre la aplicacion Ollama\n"
            "- **Linux**: Ejecuta `ollama serve`"
        )

    if _current_chain is None:
        status = switch_model(DEFAULT_SLM_MODEL)
        if "Error" in status:
            return status

    try:
        result = query_rag(_current_chain, message, _current_model)
        return format_response_with_sources(result)
    except Exception as e:
        return f"Error al generar respuesta: {e}"


EXAMPLE_QUESTIONS = [
    "Cuales son los derechos de los egresados de Uninorte?",
    "Que dice el reglamento sobre propiedad intelectual?",
    "Cuales son las obligaciones de los profesores?",
    "Que establece la politica de derechos humanos de la universidad?",
    "Cual es el reglamento interno de trabajo?",
    "Que dice la normatividad sobre bienestar universitario?",
]


def build_ui() -> gr.Blocks:
    """Construye la interfaz completa de Gradio."""
    with gr.Blocks(
        title="Asistente de Normatividad Uninorte",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Asistente de Normatividad - Universidad del Norte\n"
            "Consulta la normatividad institucional en lenguaje natural. "
            "Las respuestas se generan a partir de los documentos oficiales."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuracion")

                model_dropdown = gr.Dropdown(
                    choices=SLM_MODELS,
                    value=DEFAULT_SLM_MODEL,
                    label="Modelo SLM",
                    info="Selecciona el modelo de lenguaje",
                )
                model_status = gr.Textbox(
                    label="Estado del modelo",
                    value="Selecciona un modelo para iniciar",
                    interactive=False,
                    lines=2,
                )
                load_btn = gr.Button("Cargar modelo", variant="primary")
                status_btn = gr.Button("Verificar estado")
                status_output = gr.Textbox(
                    label="Estado del sistema",
                    interactive=False,
                    lines=4,
                )

                gr.Markdown(
                    "### Preguntas de ejemplo\n"
                    "Haz clic en una para probarla:"
                )

            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=chat_response,
                    examples=EXAMPLE_QUESTIONS,
                    title="",
                    retry_btn="Reintentar",
                    undo_btn="Deshacer",
                    clear_btn="Limpiar",
                )

        gr.Markdown(
            "---\n"
            "*Las respuestas se generan automaticamente basandose en la "
            "normatividad institucional. Consulte los documentos oficiales "
            "para informacion vinculante.*\n\n"
            "**Prototipo** - Proyecto Final SLM - Universidad del Norte"
        )

        # Event handlers
        load_btn.click(
            fn=switch_model,
            inputs=[model_dropdown],
            outputs=[model_status],
        )
        status_btn.click(
            fn=get_status,
            outputs=[status_output],
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interfaz Gradio del asistente de normatividad"
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Crear URL publica")
    args = parser.parse_args()

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
