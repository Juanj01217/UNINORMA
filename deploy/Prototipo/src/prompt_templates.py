"""Plantillas de prompts para el sistema RAG, en espanol."""

SYSTEM_PROMPT_ES = (
    "Eres UNINORMA, asistente de normatividad de la Universidad del Norte (Uninorte), Colombia. "
    "Responde usando SOLO la informacion del contexto proporcionado. "
    "Responde de forma concisa y enfocada: si la pregunta es puntual, da una respuesta directa "
    "sin copiar toda la lista del contexto. "
    "Si el contexto no tiene informacion relevante, di: "
    "'No encontre informacion sobre ese tema en los documentos disponibles.' "
    "No inventes datos. Responde en espanol."
)

RAG_PROMPT_TEMPLATE = """Fragmentos de documentos normativos de Uninorte:

{context}

{history}Pregunta: {question}

Responde solo lo que se pregunta usando la informacion de los fragmentos. Si la pregunta es concreta, da una respuesta directa y breve. Si no hay informacion relevante, indicalo."""


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


def format_history_for_prompt(history: list) -> str:
    """
    Convierte el historial de mensajes en texto para incluir en el prompt.

    history: lista de dicts con keys 'role' ('user'|'assistant') y 'content'.
    - Preguntas del usuario: se incluyen completas.
    - Respuestas del asistente: se truncan a 150 chars para evitar que el modelo
      copie respuestas largas en lugar de responder la nueva pregunta.
    Retorna string listo para insertar antes de la Pregunta actual, o '' si vacio.
    """
    if not history:
        return ""

    _MAX_ASSISTANT_CHARS = 150

    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"Pregunta previa: {content}")
        else:
            # Truncar para evitar que el modelo re-use la respuesta larga
            snippet = content[:_MAX_ASSISTANT_CHARS]
            if len(content) > _MAX_ASSISTANT_CHARS:
                snippet += "..."
            lines.append(f"Respuesta previa: {snippet}")

    if not lines:
        return ""

    return "Contexto de la conversacion:\n" + "\n".join(lines) + "\n\n"


def build_retrieval_query(question: str, history: list) -> str:
    """
    Construye una consulta de recuperacion enriquecida con contexto del historial.

    Si hay turnos previos del usuario, los concatena con la pregunta actual
    para que el retriever busque chunks relevantes para el hilo de la conversacion.
    """
    if not history:
        return question

    # Tomar los ultimos 2 turnos del usuario para enriquecer la busqueda
    user_turns = [
        m["content"].strip()
        for m in history
        if m.get("role") == "user" and m.get("content", "").strip()
    ][-2:]

    if user_turns:
        return " ".join(user_turns) + " " + question

    return question


def build_rag_prompt(context: str, question: str, history: str = "") -> str:
    """Construye el prompt completo combinando contexto, historial y pregunta."""
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        history=history,
    )
