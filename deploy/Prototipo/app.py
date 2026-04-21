"""
Interfaz Gradio para el asistente de normatividad Uninorte.

Uso:
    python app.py
    python app.py --port 7860
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
        return f"Modelo activo: {model_name}"
    except Exception as e:
        return f"Error al cargar modelo {model_name}: {e}"


def get_status() -> str:
    """Obtiene el estado actual del sistema."""
    lines = []

    if check_ollama_running():
        lines.append("Ollama: Activo")
    else:
        lines.append("Ollama: INACTIVO")
        return "\n".join(lines)

    available = get_available_models()
    installed = [m for m, v in available.items() if v]
    lines.append(f"Modelos instalados: {len(installed)}/{len(available)}")
    if installed:
        lines.append(f"  Disponibles: {', '.join(installed)}")

    if _current_model:
        lines.append(f"Modelo activo: {_current_model}")
    else:
        lines.append("Modelo activo: Ninguno")

    return "\n".join(lines)


def respond(message: str, chat_history: list):
    """Procesa mensaje del usuario y retorna respuesta para el chatbot."""
    global _current_chain, _current_model

    if not message.strip():
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "Por favor, escribe una pregunta."})
        return "", chat_history

    if not check_ollama_running():
        chat_history.append({"role": "user", "content": message})
        chat_history.append({
            "role": "assistant",
            "content": (
                "Ollama no esta activo. Inicia Ollama antes de hacer consultas.\n\n"
                "- **Windows**: Abre la aplicacion Ollama\n"
                "- **Linux**: Ejecuta `ollama serve`"
            ),
        })
        return "", chat_history

    if _current_chain is None:
        status = switch_model(DEFAULT_SLM_MODEL)
        if "Error" in status:
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": status})
            return "", chat_history

    chat_history.append({"role": "user", "content": message})

    try:
        result = query_rag(_current_chain, message, _current_model)
        answer = format_response_with_sources(result)
    except Exception as e:
        answer = f"Error al generar respuesta: {e}"

    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history


def use_example(example_text: str, chat_history: list):
    """Usa una pregunta de ejemplo."""
    return respond(example_text, chat_history)


def clear_chat():
    """Limpia el historial del chat."""
    return [], ""


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

    # Detectar modelos disponibles para el dropdown
    available = get_available_models()
    installed_models = [m for m in SLM_MODELS if available.get(m, False)]
    if not installed_models:
        installed_models = SLM_MODELS  # fallback
    default_model = installed_models[0] if installed_models else DEFAULT_SLM_MODEL

    with gr.Blocks(
        title="Asistente de Normatividad Uninorte",
    ) as app:

        gr.Markdown(
            "# Asistente de Normatividad - Universidad del Norte\n"
            "Consulta la normatividad institucional en lenguaje natural. "
            "Las respuestas se generan a partir de los documentos oficiales "
            "usando un Small Language Model (SLM) local."
        )

        with gr.Row():
            # --- Panel lateral ---
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Configuracion")

                model_dropdown = gr.Dropdown(
                    choices=installed_models,
                    value=default_model,
                    label="Modelo SLM",
                    info="Selecciona el modelo de lenguaje",
                )
                load_btn = gr.Button("Cargar modelo", variant="primary")
                model_status = gr.Textbox(
                    label="Estado del modelo",
                    value="Haz clic en 'Cargar modelo' para iniciar",
                    interactive=False,
                    lines=2,
                )

                status_btn = gr.Button("Verificar estado del sistema")
                status_output = gr.Textbox(
                    label="Estado del sistema",
                    interactive=False,
                    lines=5,
                )

                gr.Markdown("### Preguntas de ejemplo")
                for question in EXAMPLE_QUESTIONS:
                    gr.Button(
                        question[:50] + ("..." if len(question) > 50 else ""),
                        size="sm",
                    )

            # --- Area principal de chat ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    placeholder="Haz una pregunta sobre la normatividad de Uninorte...",
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Escribe tu pregunta aqui...",
                        label="",
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Enviar", variant="primary", scale=1)

                clear_btn = gr.Button("Limpiar chat", size="sm")

        gr.Markdown(
            "---\n"
            "*Las respuestas se generan automaticamente basandose en la "
            "normatividad institucional. Consulte los documentos oficiales "
            "para informacion vinculante.*\n\n"
            "**Prototipo** - Proyecto Final SLM - Universidad del Norte"
        )

        # --- Event handlers ---
        load_btn.click(
            fn=switch_model,
            inputs=[model_dropdown],
            outputs=[model_status],
        )
        status_btn.click(
            fn=get_status,
            outputs=[status_output],
        )

        # Chat: enviar con boton o Enter
        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_input],
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interfaz Gradio del asistente de normatividad"
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        theme=gr.themes.Soft(),
    )
