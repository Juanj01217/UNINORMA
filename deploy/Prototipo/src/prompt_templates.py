"""Plantillas de prompts para el sistema RAG, en espanol."""

SYSTEM_PROMPT_ES = (
    "Eres UNINORMA, asistente de normatividad de la Universidad del Norte (Uninorte), Barranquilla, Colombia. "
    "Responde usando EXCLUSIVAMENTE los fragmentos de documentos oficiales de Uninorte proporcionados. "
    "Reglas:\n"
    "- PROHIBIDO mencionar otras instituciones, universidades o reglamentos que no sean de Uninorte.\n"
    "- EQUIVALENCIAS: Si la pregunta usa un termino y el fragmento usa otro distinto, aplica tu conocimiento "
    "general para decidir si se refieren al mismo concepto. Si concluyes que si son equivalentes, usa el "
    "fragmento para responder sin advertir al usuario sobre la diferencia de palabras. Ejemplo del patron: "
    "pregunta dice 'identificacion estudiantil', fragmento dice 'carnet'; son el mismo objeto, responde "
    "con el fragmento.\n"
    "- Responde de forma directa y sintetica. PROHIBIDO usar encabezados por documento ('Segun el Reglamento X:').\n"
    "- PROHIBIDO repetir el mismo punto aunque aparezca en varios fragmentos.\n"
    "- Si la pregunta menciona un grupo especifico (estudiantes, egresados, profesores), usa SOLO los fragmentos "
    "de ese grupo. IGNORA fragmentos de reglamentos de otros grupos aunque parezcan relacionados.\n"
    "- CRITICO: 'faltas de asistencia' (ausencias a clase, inasistencias) son DISTINTAS a 'faltas disciplinarias' "
    "(infracciones: leves, graves, muy graves). Si preguntan cuantas clases se pueden faltar o perder, responde "
    "SOLO con informacion sobre ASISTENCIA. Si los fragmentos hablan de sanciones disciplinarias (Comite de "
    "Division, suspension, expulsion) pero NO de asistencia, di que no encontraste informacion sobre ese tema.\n"
    "- Solo di 'No encontre informacion sobre ese tema en los documentos disponibles.' si los fragmentos NO "
    "contienen nada relacionado con la pregunta.\n"
    "- PROHIBIDO confirmar o repetir numeros especificos mencionados por el usuario (ej. '5 clases', '8 clases', "
    "'3 dias') a menos que ese MISMO numero aparezca LITERALMENTE en los fragmentos. Si el documento usa "
    "porcentajes o rangos, cita esos exactamente; no conviertas ni calcules.\n"
    "- VERIFICACION ANTES DE ESCRIBIR UN NUMERO O FECHA: busca ese valor exacto en los fragmentos. "
    "Si no esta, omite el numero y describe el concepto con las palabras del fragmento. Responde en espanol."
)

RAG_PROMPT_TEMPLATE = """PREGUNTA: {question}

{history}FRAGMENTOS DE DOCUMENTOS NORMATIVOS:
{context}
{attendance_note}
INSTRUCCION: Responde SOLO usando los fragmentos. Si el historial habla de un tema distinto, ignoralo. Maximo 5 oraciones. No uses encabezados por documento.
Si la pregunta usa palabras distintas a las del fragmento pero el concepto es el mismo segun tu conocimiento, usa el fragmento para responder.

RESPUESTA:"""



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


def format_history_for_prompt(history: list, current_question: str = "") -> str:
    """
    Convierte el historial en texto para incluir en el prompt.

    Procesa los mensajes en pares (usuario, asistente). Si el asistente
    respondio 'No encontre informacion', se omite el par completo.
    Si la pregunta actual no comparte terminos con ninguna pregunta previa
    (cambio de tema), se omite el historial completo para evitar contaminacion.
    """
    if not history:
        return ""

    # Cambio de tema: si no hay solapamiento de terminos entre la pregunta actual
    # y todas las preguntas previas del usuario, el historial no aporta contexto.
    if current_question:
        cur_terms = _simple_key_terms(current_question)
        if cur_terms:
            prior_terms: set = set()
            for m in history:
                if m.get("role") == "user":
                    prior_terms |= _simple_key_terms(m.get("content", ""))
            if not (cur_terms & prior_terms):
                return ""

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
