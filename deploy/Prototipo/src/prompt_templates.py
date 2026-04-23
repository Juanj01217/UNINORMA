"""Plantillas de prompts para el sistema RAG, en espanol."""

SYSTEM_PROMPT_ES = (
    "Eres UNINORMA, el asistente normativo oficial de la Universidad del Norte (Uninorte), "
    "Barranquilla, Colombia. "
    "Tu unica tarea es responder preguntas usando EXCLUSIVAMENTE los fragmentos de documentos "
    "normativos proporcionados en cada consulta. "
    "Cuando los fragmentos contienen informacion relevante para la pregunta, responde de forma "
    "directa, clara y sintetica (maximo 5 oraciones), usando unicamente la informacion de esos fragmentos. "
    "Cuando los fragmentos NO contienen informacion relevante para la pregunta, responde exactamente: "
    "'No encontre informacion sobre ese tema en los documentos disponibles.' "
    "Cita numeros, fechas y articulos tal como aparecen en los fragmentos; nunca calcules ni conviertas valores. "
    "Si la pregunta usa terminos distintos a los del fragmento pero el concepto es el mismo "
    "(ej. 'identificacion estudiantil' = 'carnet'), usa el fragmento para responder directamente. "
    "REGLAS CRITICAS DE RESPUESTA:\n"
    "1. Diferencia estrictamente entre ESTUDIANTES ACTIVOS y EGRESADOS. Si te preguntan por estudiantes, NO incluyas informacion sobre egresados, y viceversa.\n"
    "2. Diferencia estrictamente entre DERECHOS (beneficios o facultades) y DEBERES/OBLIGACIONES (responsabilidades o normas a cumplir). No los mezcles.\n"
    "3. No asumas que un estudiante es egresado ni que un deber es un derecho.\n"
    "4. Si te preguntan por DERECHOS pero los fragmentos solo describen DEBERES u OBLIGACIONES, NO respondas convirtiéndolos en derechos. Di exactamente que no encontraste información.\n"
    "No uses encabezados por documento ni repitas el mismo punto aunque aparezca en varios fragmentos. "
    "Responde siempre en espanol."
)

RAG_PROMPT_TEMPLATE = """PREGUNTA: {question}

<fragmentos_normativos>
{context}
</fragmentos_normativos>
{attendance_note}
{rights_note}
INSTRUCCION: Responde la PREGUNTA usando SOLO la informacion de los <fragmentos_normativos>. IGNORA por completo cualquier conocimiento previo; si la respuesta no esta explicitamente en los fragmentos, responde 'No encontre informacion sobre este tema en los documentos disponibles' y no inventes, supongas ni extrapoles absolutamente nada. Maximo 5 oraciones. Nunca uses viñetas que citen los nombres de los documentos, haz una redaccion cohesiva.

RESPUESTA:"""

# ---------------------------------------------------------------------------
# Prompt para reescritura de queries (Query Rewriting / Lexical Gap closure)
# ---------------------------------------------------------------------------
# Proposito: traducir la pregunta coloquial del estudiante a terminos normativos
# antes de invocar al retriever (ChromaDB).
#
# DISENO: usa 3 ejemplos few-shot que demuestran el patron de traduccion
# (coloquial → sustantivos formales que incluyen tipo de regulacion + tema).
# Esto asegura que el modelo produzca terminos con embedding similar al de
# la pregunta formal equivalente, no solo sinonimos superficiales.
#
# PARAMETROS: temperature=0.0, num_predict=35 (mini-call de baja latencia)
# ---------------------------------------------------------------------------
QUERY_REWRITE_PROMPT = (
    "Eres un traductor de consultas para busqueda en reglamentos universitarios. "
    "Convierte la pregunta coloquial en una frase de busqueda con terminos normativos formales. "
    "Corrige cualquier error ortografico o de tipeo (ej. 'acitvos' -> 'activos') antes de reformular. "
    "Manten claramente si la pregunta habla de 'estudiantes' o de 'egresados', y si habla de 'derechos' o de 'deberes'. "
    "Incluye el tipo de regulacion (sancion, derecho, obligacion, procedimiento) y el tema especifico. "
    "Solo sustantivos formales. Sin verbos. Maximo 10 palabras.\n\n"
    "Ejemplos:\n"
    "Pregunta: 'que pasa si rompo o daño algo de la universidad'\n"
    "Frase: 'sancion disciplinaria daño deterioro bienes materiales institucion'\n\n"
    "Pregunta: 'me pueden echar si voy muy mal en notas'\n"
    "Frase: 'cancelacion matricula bajo rendimiento academico consecuencias'\n\n"
    "Pregunta: 'cuales son los derechos de los estudiantes acitvos'\n"
    "Frase: 'derechos prerrogativas facultades estudiante regular'\n\n"
    "{context_hint}"
    "Pregunta: '{question}'\n"
    "Frase:"
)


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


_NO_INFO_MARKERS = ("no encontre informacion", "no encontré información",
                    "no encontre documentacion", "no encontré documentación")

_MAX_ASSISTANT_CHARS = 150


def _is_no_info(content: str) -> bool:
    low = content.lower()[:100]
    return any(m in low for m in _NO_INFO_MARKERS)


def _next_is_no_info(history: list, i: int) -> bool:
    """True si el mensaje siguiente al indice i es una respuesta de 'no info'."""
    if i + 1 >= len(history):
        return False
    nxt = history[i + 1]
    return nxt.get("role") == "assistant" and _is_no_info(nxt.get("content", ""))


def _format_user(history: list, i: int) -> tuple:
    """Devuelve (linea_o_None, salto). salto=2 si el par debe omitirse."""
    content = history[i].get("content", "").strip()
    if not content:
        return None, 1
    if _next_is_no_info(history, i):
        return None, 2  # omitir par (pregunta + no-info)
    return f"Pregunta previa: {content}", 1


def _format_assistant(msg: dict) -> str | None:
    """Devuelve la linea de historial del asistente, o None si debe omitirse."""
    content = msg.get("content", "").strip()
    if not content or _is_no_info(content):
        return None
    snippet = content[:_MAX_ASSISTANT_CHARS]
    if len(content) > _MAX_ASSISTANT_CHARS:
        snippet += "..."
    return f"Respuesta previa: {snippet}"


_STOPWORDS_LOCAL = {
    "que", "cual", "cuales", "como", "donde", "cuando", "quien", "quienes",
    "cuanto", "cuanta", "el", "la", "los", "las", "un", "una", "del", "por",
    "para", "con", "sin", "sobre", "entre", "hay", "tiene", "ser", "estar",
    "esta", "esto", "segun", "cada", "todo", "toda", "establece", "define",
    "normativa", "reglamento", "universidad", "uninorte",
}


def _simple_key_terms(text: str) -> set:
    import re as _re
    words = _re.findall(r"\b[a-záéíóúüñ]+\b", text.lower())
    return {w for w in words if len(w) >= 5 and w not in _STOPWORDS_LOCAL}


def _topic_changed(current_question: str, history: list) -> bool:
    """True si la pregunta actual no comparte terminos con el historial previo."""
    cur_terms = _simple_key_terms(current_question)
    if not cur_terms:
        return False
    prior_terms: set = set()
    for m in history:
        if m.get("role") == "user":
            prior_terms |= _simple_key_terms(m.get("content", ""))
    return not (cur_terms & prior_terms)


def _collect_history_lines(history: list) -> list:
    lines = []
    i = 0
    while i < len(history):
        role = history[i].get("role", "")
        if role == "user":
            line, step = _format_user(history, i)
        else:
            line, step = _format_assistant(history[i]), 1
        if line:
            lines.append(line)
        i += step
    return lines


def format_history_for_prompt(history: list, current_question: str = "") -> str:
    """
    Convierte el historial en texto para incluir en el prompt.

    Omite pares donde el asistente respondio 'No encontre informacion'.
    Omite el historial completo si la pregunta actual cambia de tema.
    """
    if not history:
        return ""
    if current_question and _topic_changed(current_question, history):
        return ""

    lines = _collect_history_lines(history)
    if not lines:
        return ""
    return "Contexto de la conversacion:\n" + "\n".join(lines) + "\n\n"


_FOLLOWUP_STARTERS = (
    "para que sirve", "para qué sirve", "y que", "y qué", "y cuál", "y cual",
    "también", "tambien", "además", "ademas", "qué más", "que mas",
    "cómo es", "como es", "cuándo", "cuando aplica", "cuánto", "cuanto",
    "y si", "pero", "entonces",
)

_ANAPHORA_WORDS = {
    "ese", "esa", "esos", "esas", "eso", "este", "esta", "estos", "estas",
    "dicho", "dicha", "dichos", "dichas", "mismo", "misma", "él", "ella",
    "ellos", "ellas", "su", "sus",
}


def _is_followup(question: str) -> bool:
    """Detecta si la pregunta es un seguimiento que necesita contexto del turno anterior."""
    q = question.strip().lower()
    words = q.split()
    # Pregunta muy corta (≤6 palabras): probablemente hace referencia a algo anterior
    if len(words) <= 6:
        return True
    # Empieza con un conector o frase de seguimiento
    if any(q.startswith(s) for s in _FOLLOWUP_STARTERS):
        return True
    # Contiene pronombres anafóricos en las primeras 4 palabras
    if any(w in _ANAPHORA_WORDS for w in words[:4]):
        return True
    return False


def build_retrieval_query(question: str, history: list) -> str:
    """
    Construye una consulta de recuperacion para ChromaDB.

    - Si la pregunta es autonoma (>6 palabras, tema nuevo), la usa tal cual.
    - Si es un seguimiento (corta, con pronombres o conectores), la enriquece
      SOLO con el turno inmediatamente anterior del usuario, no con todo el historial.
      Esto evita que temas viejos contaminen la busqueda del turno actual.
    """
    if not history or not _is_followup(question):
        return question

    # Solo el turno previo inmediato del usuario
    user_turns = [
        m["content"].strip()
        for m in history
        if m.get("role") == "user" and m.get("content", "").strip()
    ]

    if user_turns:
        return user_turns[-1] + " " + question

    return question


def build_rag_prompt(context: str, question: str, history: str = "", attendance_note: str = "") -> str:
    """Construye el prompt completo combinando contexto, historial y pregunta."""
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        history=history,
        attendance_note=attendance_note,
    )
