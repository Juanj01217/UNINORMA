"""Plantillas de prompts para el sistema RAG, en espanol."""

SYSTEM_PROMPT_ES = (
    "Eres UNINORMA, un asistente virtual de la Universidad del Norte (Uninorte), "
    "Barranquilla, Colombia. Ayudas a estudiantes, profesores y empleados "
    "a entender la normatividad institucional.\n\n"
    "Instrucciones:\n"
    "1. Usa la informacion del contexto proporcionado para responder.\n"
    "2. Responde en espanol de manera clara, organizada y util.\n"
    "3. Cuando cites informacion, menciona el documento fuente.\n"
    "4. Si el contexto contiene informacion relacionada, usala para dar la mejor respuesta posible.\n"
    "5. Solo di que no tienes informacion si el contexto realmente no tiene NADA relacionado con la pregunta."
)

RAG_PROMPT_TEMPLATE = """A continuacion se presentan fragmentos de documentos normativos de la Universidad del Norte:

{context}

Pregunta: {question}

Responde la pregunta usando la informacion de los fragmentos anteriores. Menciona de que documento proviene la informacion. Organiza tu respuesta de forma clara."""


def format_context_from_docs(docs: list) -> str:
    """
    Formatea documentos recuperados en un string de contexto con etiquetas de fuente.

    Cada chunk se prefija con [Fuente: filename, Pagina: N].
    """
    context_parts = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        source = metadata.get("source", "Desconocido")
        page = metadata.get("page", "?")
        title = metadata.get("title", "")

        header = f"[Fuente: {source} | Pagina: {page}]"
        if title:
            header = f"[Fuente: {title} ({source}) | Pagina: {page}]"

        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(context_parts)


def build_rag_prompt(context: str, question: str) -> str:
    """Construye el prompt completo combinando contexto y pregunta."""
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)
