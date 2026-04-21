"""Cadena RAG: combina retriever + LLM para Q&A sobre normatividad."""
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


def _dedup_answer(text: str) -> str:
    """
    Elimina lineas y bullets duplicados de la respuesta del LLM.
    Tambien elimina secciones repetidas del tipo 'Segun el X, Y son:'.
    """
    lines = text.splitlines()
    seen: set = set()
    result: list = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip()).lower()
        if normalized and normalized in seen:
            continue
        if normalized:
            seen.add(normalized)
        result.append(line)
    return "\n".join(result)


_NO_INFO = (
    "No encontre documentacion especifica de Uninorte sobre este tema. "
    "Para orientacion, te recomendamos consultar directamente con Bienestar Universitario "
    "o la dependencia correspondiente de la Universidad del Norte."
)

_STOPWORDS_ES = {
    "que", "cual", "cuales", "como", "donde", "cuando", "quien", "quienes",
    "cuanto", "cuanta", "cuantos", "cuantas",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "por",
    "para", "con", "sin", "sobre", "entre", "hasta", "desde", "hay",
    "tiene", "tienen", "tienes", "ser", "estar", "pasa", "sirve", "dice",
    "esta", "esto", "estos", "estas", "segun", "cada", "todo", "toda",
    "establece", "define", "menciona", "explica", "debo", "puedo",
    # Adverbios de precision/cantidad que no aparecen en reglamentos y
    # bajan el score coseno del retrieval query sin aportar semantica normativa.
    "exactamente", "aproximadamente", "solamente", "unicamente", "exacto",
    "concretamente", "especificamente", "precisamente",
    # Verbos conversacionales: no aparecen en documentos normativos y
    # alejan el embedding del retrieval query del espacio de los reglamentos.
    "hablame", "cuentame", "dime", "quiero", "busco", "consigo",
    "obtengo", "necesitas", "podrias",
}

# Palabras que aparecen en casi todos los chunks normativos — no sirven como
# discriminadores porque su presencia no confirma que el chunk responde la pregunta.
_OMNIPRESENT = {
    "normativa", "reglamento", "universidad", "uninorte",
    "conforme", "mediante", "disposicion", "articulo",
}

_ATTENDANCE_KEYWORDS = {"falto", "faltar", "faltas", "clase", "clases", "asistencia", "inasistencia", "ausencia"}

# Art. 70 Reglamento Estudiantil: >25% de inasistencias = 0.0 para pregrado.
# Art. 73: >20% para lenguas extranjeras.
# Semestre tipico de pregrado: ~48 sesiones (3h/semana × 16 semanas).
_ATTENDANCE_DEFAULT_SESSIONS = 48
_ATTENDANCE_THRESHOLD_PCT = 0.25      # Art. 70 — pregrado general
_ATTENDANCE_LANG_THRESHOLD_PCT = 0.20  # Art. 73 — lenguas extranjeras

_NUMBER_WORDS_ES = {
    "una": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20,
}

_ATTENDANCE_RULE_NOTE = (
    "[REGLA EXACTA Art. 70 Reglamento Estudiantil — usa esto para responder]: "
    "La sancion de 0.0 solo aplica cuando las faltas EXCEDAN el 25% del total de clases del periodo. "
    "En un semestre tipico de pregrado (~48 sesiones), eso equivale a MAS DE 12 clases. "
    "Para lenguas extranjeras el umbral es el 20% (Art. 73).\n"
)


def _is_attendance_question(question: str) -> bool:
    words = set(re.findall(r"\b[a-záéíóúüñ]+\b", question.lower()))
    return len(words & _ATTENDANCE_KEYWORDS) >= 2


def _is_language_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ("lengua", "idioma", "ingles", "frances", "aleman", "portugues", "mandarin"))


def _extract_absence_count(question: str) -> Optional[int]:
    """Extrae el numero de clases/faltas mencionado en la pregunta, o None."""
    q = question.lower()
    m = re.search(r"\b(\d{1,3})\b", q)
    if m:
        n = int(m.group(1))
        if 0 < n < 200:
            return n
    for word, val in _NUMBER_WORDS_ES.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            return val
    return None


def _deterministic_attendance_answer(question: str) -> Optional[str]:
    """
    Cuando la pregunta es sobre asistencia y menciona un numero especifico,
    responde deterministicamente usando la regla del Art. 70/73, sin invocar el LLM.
    El LLM no aplica la condicion 'excedan el 25%' correctamente con modelos pequenos.
    """
    if not _is_attendance_question(question):
        return None
    q_lower = question.lower()
    is_lang = _is_language_question(question)
    pct = _ATTENDANCE_LANG_THRESHOLD_PCT if is_lang else _ATTENDANCE_THRESHOLD_PCT
    threshold_int = int(_ATTENDANCE_DEFAULT_SESSIONS * pct)  # 12 o 9
    article = "Art. 73" if is_lang else "Art. 70"
    pct_label = f"{int(pct * 100)}%"

    n = _extract_absence_count(question)

    if n is None:
        # Pregunta por el umbral mismo ("cuantas faltas", "maximo de faltas")
        asking_threshold = any(
            w in q_lower
            for w in ("cuantas", "cuántas", "cuantos", "maximo", "máximo",
                      "limite", "límite", "permitido", "permitidas", "necesito")
        )
        if not asking_threshold:
            return None
        return (
            f"Segun el {article} del Reglamento Estudiantil de Uninorte, las faltas de asistencia deben "
            f"EXCEDER el {pct_label} del total de clases programadas en el periodo para perder el derecho "
            f"al examen final con calificacion 0.0. En un semestre tipico de pregrado con "
            f"{_ATTENDANCE_DEFAULT_SESSIONS} sesiones, eso equivale a mas de {threshold_int} clases. "
            f"El numero exacto depende del total de sesiones de tu asignatura segun el silabo."
        )

    if n > threshold_int:
        return (
            f"Con {n} faltas en un semestre tipico de {_ATTENDANCE_DEFAULT_SESSIONS} sesiones "
            f"SI superas el umbral del {pct_label} ({article} del Reglamento Estudiantil), "
            f"por lo que la asignatura se calificaria con 0.0 (cero punto cero). "
            f"El umbral exacto depende del numero total de sesiones programadas en tu asignatura."
        )
    return (
        f"Con {n} faltas en un semestre tipico de {_ATTENDANCE_DEFAULT_SESSIONS} sesiones "
        f"NO superas el umbral del {pct_label} ({article} del Reglamento Estudiantil), "
        f"que equivale a mas de {threshold_int} inasistencias. "
        f"Con {n} faltas no perderas el derecho al examen final por inasistencia, "
        f"aunque el umbral exacto depende del total de sesiones de tu asignatura."
    )


def _key_terms(question: str) -> set:
    """
    Extrae terminos discriminadores de la pregunta.
    Excluye stopwords Y palabras omnipresentes en el corpus (que aparecen en casi
    todos los chunks y por tanto no sirven para detectar mismatches semanticos).
    """
    words = re.findall(r"\b[a-záéíóúüñ]+\b", question.lower())
    return {
        w for w in words
        if len(w) >= 5
        and w not in _STOPWORDS_ES
        and w not in _OMNIPRESENT
    }


def _token_in_text(term: str, text: str) -> bool:
    """Busqueda con tolerancia a plurales/genero comunes del espanol."""
    pattern = r"\b" + re.escape(term) + r"(s|es|a|as|os|ados|idas|idos)?\b"
    return bool(re.search(pattern, text))


def _context_covers(question: str, docs: list) -> bool:
    """
    Verifica que al menos un termino clave de la pregunta aparezca en los chunks.

    Reglas:
    - 0 terminos + pregunta corta (≤ 3 palabras): bloquear (saludos, comandos).
    - 1+ terminos: al menos uno debe aparecer en el contexto recuperado.
      Incluir el caso de 1 termino (antes era bypass) previene hallucinations
      cuando el retriever devuelve chunks sin el termino clave de la pregunta
      (ej. "carnet" en pregunta pero chunks de Res_Con_Aca que no lo mencionan).
    """
    terms = _key_terms(question)
    if len(terms) == 0:
        return len(question.strip().split()) > 3
    context_text = " ".join(doc.page_content.lower() for doc in docs)
    return any(_token_in_text(t, context_text) for t in terms)


_NON_NORMATIVE_KW = ("informe", "sostenibilidad", "sustainability", "memoria")

_ACRONYM_RE = re.compile(r"\b[A-ZÁÉÍÓÚ]{3,}\b")
_ALLOWED_ACRONYMS = {"UNINORTE", "PDF", "DNI", "NRC", "TIC", "GPS", "URL", "API", "ID"}


def _validate_no_invented_acronyms(answer: str, docs: List[Document]) -> str:
    """Elimina del texto acronimos de 3+ mayusculas que no aparecen en los chunks fuente."""
    corpus = " ".join(d.page_content for d in docs)
    found = set(_ACRONYM_RE.findall(answer)) - _ALLOWED_ACRONYMS
    invented = [a for a in found if a not in corpus]
    if not invented:
        return answer
    for a in invented:
        answer = re.sub(rf"\b{re.escape(a)}\b", "[termino no verificado]", answer)
    return answer


sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, DEFAULT_SLM_MODEL, TEMPERATURE, MAX_TOKENS
from src.prompt_templates import (
    SYSTEM_PROMPT_ES,
    RAG_PROMPT_TEMPLATE,
    format_context_from_docs,
    format_history_for_prompt,
)


def create_llm(
    model_name: str = DEFAULT_SLM_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Ollama:
    """Crea una instancia de LLM Ollama para LangChain."""
    return Ollama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=max_tokens,
        system=SYSTEM_PROMPT_ES,
    )


class RAGChain:
    """Cadena RAG que encapsula retriever + prompt + LLM con soporte de historial."""

    def __init__(self, retriever, llm, prompt: PromptTemplate):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Formatea documentos recuperados en un string de contexto."""
        return format_context_from_docs(docs)

    @staticmethod
    def _filter_and_dedup(docs: List[Document], seen: set) -> List[Document]:
        result = []
        for doc in docs:
            title = doc.metadata.get("title", "").lower()
            source = doc.metadata.get("source", "").lower()
            if any(kw in title or kw in source for kw in _NON_NORMATIVE_KW):
                continue
            fingerprint = doc.page_content[:120].strip()
            if fingerprint not in seen:
                seen.add(fingerprint)
                result.append(doc)
        return result

    def _keyword_search_fallback(self, terms: set, n: int = 3) -> List[Document]:
        """
        Busca chunks que contengan literalmente algun termino clave usando filtrado
        por contenido (where_document), sin requerir computacion de embeddings.
        Se usa cuando la busqueda vectorial retorna chunks que no mencionan
        el concepto de la pregunta (mismatch semantico detectado por _context_covers).
        """
        try:
            collection = self.retriever.vectorstore._collection
            for term in sorted(terms, key=len, reverse=True):
                if len(term) < 4:
                    continue
                result = collection.get(
                    where_document={"$contains": term},
                    include=["documents", "metadatas"],
                    limit=n,
                )
                raw_docs = result.get("documents") or []
                raw_metas = result.get("metadatas") or []
                if raw_docs:
                    return [
                        Document(page_content=text, metadata=meta or {})
                        for text, meta in zip(raw_docs, raw_metas)
                    ]
        except Exception:
            pass
        return []

    def invoke(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta la cadena RAG y retorna respuesta + documentos fuente.

        Args:
            question: Pregunta actual del usuario.
            history: Lista de dicts [{role, content}] con turnos previos
                     (max ~6 mensajes = 3 exchanges). Puede ser None o [].

        Flujo:
            1. Enriquecer la query con contexto del historial para mejor retrieval.
            2. Recuperar chunks relevantes.
            3. Formatear contexto e historial en el prompt.
            4. Invocar el LLM.
        """
        det = _deterministic_attendance_answer(question)
        if det is not None:
            _, source_docs, _ = self._prepare(question, history)
            return {"answer": det, "source_documents": source_docs}

        prompt_text, source_docs, _ = self._prepare(question, history)
        if not source_docs:
            return {"answer": _NO_INFO, "source_documents": []}
        raw_answer = self.llm.invoke(prompt_text)
        answer = raw_answer.content if hasattr(raw_answer, "content") else str(raw_answer)
        answer = _dedup_answer(answer)
        answer = _validate_no_invented_acronyms(answer, source_docs)
        return {
            "answer": answer,
            "source_documents": source_docs,
        }

    def _prepare(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]],
    ):
        """Recupera docs, construye prompt y sources_info. Compartido por invoke e invoke_stream."""
        if history is None:
            history = []

        # Usar solo los terminos clave como query de recuperacion mejora el score
        # coseno para preguntas informales: "falto exactamente 8 clases" tiene baja
        # similitud con el texto formal del reglamento, pero "falto clases materia"
        # (terminos clave) si coincide bien con los chunks de asistencia.
        terms = _key_terms(question)
        retrieval_query = " ".join(terms) if terms else question
        source_docs = self.retriever.invoke(retrieval_query)
        seen_fingerprints: set = set()
        unique_docs = self._filter_and_dedup(source_docs, seen_fingerprints)

        needs_fallback = (
            (unique_docs and not _context_covers(question, unique_docs))
            or (not unique_docs and bool(terms))
        )
        if needs_fallback:
            fallback = self._keyword_search_fallback(terms)
            unique_docs = self._filter_and_dedup(fallback, seen_fingerprints)

        context = self._format_docs(unique_docs)
        history_text = format_history_for_prompt(history, current_question=question)
        attendance_note = _ATTENDANCE_RULE_NOTE if _is_attendance_question(question) else ""
        prompt_text = self.prompt.format(
            context=context,
            question=question,
            history=history_text,
            attendance_note=attendance_note,
        )

        seen_sources: set = set()
        sources_info = []
        for doc in unique_docs:
            meta = doc.metadata
            key = (meta.get("source", ""), str(meta.get("page", "")))
            if key not in seen_sources:
                seen_sources.add(key)
                sources_info.append({
                    "source": meta.get("source", ""),
                    "title": meta.get("title", ""),
                    "page": str(meta.get("page", "")),
                })

        return prompt_text, unique_docs, sources_info

    def invoke_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """Ejecuta la cadena RAG y devuelve (sources_info, source_docs, token_generator)."""
        det = _deterministic_attendance_answer(question)
        if det is not None:
            _, source_docs, sources_info = self._prepare(question, history)
            def _det_gen():
                yield det
            return sources_info, source_docs, _det_gen()

        prompt_text, source_docs, sources_info = self._prepare(question, history)
        if not source_docs:
            def _no_info_gen():
                yield _NO_INFO
            return [], [], _no_info_gen()
        return sources_info, source_docs, self.llm.stream(prompt_text)


def create_rag_chain(
    retriever,
    model_name: str = DEFAULT_SLM_MODEL,
    temperature: float = TEMPERATURE,
) -> RAGChain:
    """Construye la cadena RAG completa: retriever -> prompt -> LLM -> respuesta."""
    llm = create_llm(model_name, temperature)

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question", "history", "attendance_note"],
    )

    return RAGChain(retriever, llm, prompt)


def query_rag(
    chain: RAGChain,
    question: str,
    model_name: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Ejecuta una consulta RAG y retorna el resultado estructurado.

    Args:
        chain: Instancia RAGChain ya configurada.
        question: Pregunta del usuario.
        model_name: Nombre del modelo (para incluir en el resultado).
        history: Historial de la conversacion [{role, content}].

    Returns:
        Diccionario con: answer, source_documents, sources_info, model.
    """
    result = chain.invoke(question, history=history)

    source_docs = result.get("source_documents", [])
    sources_info = []
    for doc in source_docs:
        meta = doc.metadata
        sources_info.append({
            "source": meta.get("source", ""),
            "title": meta.get("title", ""),
            "page": meta.get("page", ""),
        })

    return {
        "answer": result.get("answer", "Sin respuesta"),
        "source_documents": source_docs,
        "sources_info": sources_info,
        "model": model_name,
    }


def format_response_with_sources(result: Dict[str, Any]) -> str:
    """Formatea la respuesta con citas de fuentes para mostrar al usuario."""
    answer = result["answer"]
    sources = result.get("sources_info", [])

    if not sources:
        return answer

    # Eliminar fuentes duplicadas
    seen = set()
    unique_sources = []
    for s in sources:
        key = (s["source"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    sources_text = "\n\n---\n**Fuentes consultadas:**\n"
    for s in unique_sources:
        title = s.get("title", s["source"])
        page = s.get("page", "?")
        sources_text += f"- {title} (pag. {page})\n"

    return answer + sources_text
