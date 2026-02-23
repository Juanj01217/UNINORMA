"""Plantillas de prompts para el sistema RAG, en espanol."""

SYSTEM_PROMPT_ES = (
    "Eres un asistente virtual especializado en la normatividad "
    "institucional de la Universidad del Norte (Uninorte), Barranquilla, Colombia. "
    "Tu funcion es responder preguntas basandote EXCLUSIVAMENTE en los fragmentos "
    "de documentos normativos que se te proporcionan como contexto.\n\n"
    "Reglas estrictas:\n"
    "1. Responde UNICAMENTE con informacion que aparezca en el contexto proporcionado.\n"
    "2. Si la respuesta no se encuentra en el contexto, di claramente: "
    "\"No encontre informacion sobre este tema en la normatividad disponible.\"\n"
    "3. Cita el documento fuente y, si es posible, el articulo o seccion relevante.\n"
    "4. Responde en espanol de manera clara y concisa.\n"
    "5. No inventes informacion ni hagas suposiciones fuera del contexto dado."
)

RAG_PROMPT_TEMPLATE = """Contexto de normatividad institucional:
---
{context}
---

Pregunta del usuario: {question}

Instrucciones: Responde la pregunta basandote exclusivamente en el contexto proporcionado. Indica el documento fuente de la informacion. Si no encuentras la respuesta en el contexto, indica que no tienes esa informacion."""


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
