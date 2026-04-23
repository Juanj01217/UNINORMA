"""Cadena RAG: combina retriever + LLM para Q&A sobre normatividad."""
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

_logger = logging.getLogger(__name__)


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
    # Adverbios de precision/cantidad
    "exactamente", "aproximadamente", "solamente", "unicamente", "exacto",
    "concretamente", "especificamente", "precisamente",
    # Verbos conversacionales
    "hablame", "cuentame", "dime", "quiero", "busco", "consigo",
    "obtengo", "necesitas", "podrias",
}

# Palabras omnipresentes en el corpus — no sirven como discriminadores.
_OMNIPRESENT = {
    "normativa", "reglamento", "universidad", "uninorte",
    "conforme", "mediante", "disposicion", "articulo",
}

_ATTENDANCE_KEYWORDS = {"falto", "faltar", "faltas", "clase", "clases", "asistencia", "inasistencia", "ausencia"}

_ATTENDANCE_DEFAULT_SESSIONS = 48
_ATTENDANCE_THRESHOLD_PCT = 0.25
_ATTENDANCE_LANG_THRESHOLD_PCT = 0.20

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
    Responde deterministicamente preguntas de asistencia con numero especifico.
    Evita que el LLM malinterprete la condicion '>25%' del Art. 70/73.
    """
    if not _is_attendance_question(question):
        return None
    q_lower = question.lower()
    is_lang = _is_language_question(question)
    pct = _ATTENDANCE_LANG_THRESHOLD_PCT if is_lang else _ATTENDANCE_THRESHOLD_PCT
    threshold_int = int(_ATTENDANCE_DEFAULT_SESSIONS * pct)
    article = "Art. 73" if is_lang else "Art. 70"
    pct_label = f"{int(pct * 100)}%"

    n = _extract_absence_count(question)

    if n is None:
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
    Excluye stopwords y palabras omnipresentes en el corpus.
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


def _context_covers(
    question: str,
    docs: list,
    threshold: float = 0.35,
    rewritten_query: str = "",
) -> bool:
    """
    Verifica que al menos el `threshold` (35%) de los terminos clave aparezcan
    en los chunks recuperados.

    Evalua la UNION de terminos de la pregunta original Y del rewrite.
    Esto resuelve el 'Lexical Gap paradox':
    - Sin rewrite: evalua solo la pregunta original (comportamiento base).
    - Con rewrite: la pregunta coloquial ("laptop", "descompongo") puede no tener
      sus terminos en los chunks formales, PERO el rewrite ("sanciones", "bienes")
      SI los tiene. La union evita rechazar chunks correctamente recuperados.
    """
    terms = _key_terms(question)
    if rewritten_query:
        terms = terms | _key_terms(rewritten_query)

    if len(terms) == 0:
        return len(question.strip().split()) > 3
    context_text = " ".join(doc.page_content.lower() for doc in docs)
    matched = sum(1 for t in terms if _token_in_text(t, context_text))
    return (matched / len(terms)) >= threshold


_NON_NORMATIVE_KW = ("informe", "sostenibilidad", "sustainability", "memoria")

_ACRONYM_RE = re.compile(r"\b[A-ZÁÉÍÓÚ]{3,}\b")
_ALLOWED_ACRONYMS = {"UNINORTE", "PDF", "DNI", "NRC", "TIC", "GPS", "URL", "API", "ID"}


def _validate_no_invented_acronyms(answer: str, docs: List[Document]) -> str:
    """Elimina acronimos inventados que no aparecen en los chunks fuente."""
    corpus = " ".join(d.page_content for d in docs)
    found = set(_ACRONYM_RE.findall(answer)) - _ALLOWED_ACRONYMS
    invented = [a for a in found if a not in corpus]
    if not invented:
        return answer
    for a in invented:
        answer = re.sub(rf"\b{re.escape(a)}\b", "[termino no verificado]", answer)
    return answer


# ---------------------------------------------------------------------------
# Query Rewriting — validacion del output del LLM rewriter
# ---------------------------------------------------------------------------
_REWRITE_MAX_CHARS = 150

_REWRITE_BAD_PREFIXES = (
    "no ", "lo siento", "disculpa", "lo que pregunta",
    "la pregunta", "respuesta:", "segun ", "según ",
    "como asistente", "no puedo", "estimado", "hola",
)


def _is_bad_rewrite(rewritten: str, original: str) -> bool:
    """
    True si el output del LLM rewriter es invalido:
    - Vacio o muy corto (< 4 chars)
    - Demasiado largo (> 150 chars — modelo ignoro 'maximo 10 palabras')
    - Empieza con prefijo de respuesta o disculpa
    """
    cleaned = rewritten.strip()
    if not cleaned or len(cleaned) < 4:
        return True
    if len(cleaned) > _REWRITE_MAX_CHARS:
        return True
    low = cleaned.lower()
    return any(low.startswith(p) for p in _REWRITE_BAD_PREFIXES)


sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, DEFAULT_SLM_MODEL, TEMPERATURE, MAX_TOKENS
from src.prompt_templates import (
    SYSTEM_PROMPT_ES,
    RAG_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT,
    format_context_from_docs,
    format_history_for_prompt,
)


def create_llm(
    model_name: str = DEFAULT_SLM_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Ollama:
    """LLM principal para generar la respuesta final al usuario."""
    return Ollama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=max_tokens,
        system=SYSTEM_PROMPT_ES,
        repeat_penalty=1.15,
    )


def create_rewrite_llm(model_name: str = DEFAULT_SLM_MODEL) -> Ollama:
    """
    LLM dedicado exclusivamente a Query Rewriting.

    Parametros optimizados para velocidad y salida corta:
    - temperature=0.0  -> greedy decoding, determinista
    - num_predict=35   -> suficiente para 10 palabras con tokens BPE en espanol
    - repeat_penalty=1.0 -> sin penalizacion (innecesaria en salidas cortas)
    - Sin system prompt -> el QUERY_REWRITE_PROMPT es autocontenido
    """
    return Ollama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=35,
        repeat_penalty=1.0,
    )


class RAGChain:
    """Cadena RAG que encapsula retriever + prompt + LLM con soporte de historial."""

    def __init__(
        self,
        retriever,
        llm,
        prompt: PromptTemplate,
        rewrite_llm: Optional[Ollama] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.rewrite_llm = rewrite_llm

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return format_context_from_docs(docs)

    @staticmethod
    def _filter_and_dedup(docs: List[Document], seen: set, question: str = "") -> List[Document]:
        result = []
        q_lower = question.lower()
        is_student_query = "estudiant" in q_lower
        is_alumni_query = "egresad" in q_lower

        for doc in docs:
            title = doc.metadata.get("title", "").lower()
            source = doc.metadata.get("source", "").lower()
            
            # Exclusion basica por palabras clave no normativas
            if any(kw in title or kw in source for kw in _NON_NORMATIVE_KW):
                continue
                
            # Validacion cruzada estricta: si preguntan por estudiantes, omitir doc egresados, y viceversa
            is_alumni_doc = "egresado" in title or "egresado" in source
            is_student_doc = "estudiant" in title or "estudiant" in source
            
            if is_student_query and not is_alumni_query and is_alumni_doc:
                continue
            if is_alumni_query and not is_student_query and is_student_doc:
                continue

            fingerprint = doc.page_content[:120].strip()
            if fingerprint not in seen:
                seen.add(fingerprint)
                result.append(doc)
        return result

    def _keyword_search_fallback(self, terms: set, n: int = 3) -> List[Document]:
        """Busqueda literal por keyword en ChromaDB cuando el vector search falla."""
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

    def _rewrite_query_for_retrieval(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        """
        Traduce la pregunta coloquial a terminos normativos formales para ChromaDB.

        Usa few-shot examples en el prompt para garantizar que el output tenga
        embedding similar al de la pregunta formal equivalente. Esto es clave:
        no basta con sinonimos superficiales, los terminos deben estar en el
        mismo espacio semantico que los reglamentos indexados.

        Ejemplos de transformacion (guiados por los few-shots):
          "descompongo un laptop prestado"
            -> "sancion disciplinaria daño deterioro bienes materiales institucion"

          "me pueden echar si voy muy mal"
            -> "cancelacion matricula bajo rendimiento academico consecuencias"

        Fallback en 3 niveles si el rewrite falla:
          1. LLM rewrite (este metodo)
          2. _key_terms(question) — extraccion heuristica
          3. question cruda
        """
        terms = _key_terms(question)
        fallback = " ".join(terms) if terms else question

        if self.rewrite_llm is None:
            return fallback

        # Enriquecer preguntas de seguimiento (cortas / con anafora) con el
        # turno previo del usuario para que el rewrite tenga contexto suficiente.
        context_hint = ""
        if history:
            user_turns = [
                m["content"].strip()
                for m in history
                if m.get("role") == "user" and m.get("content", "").strip()
            ]
            if user_turns and len(question.split()) <= 8:
                context_hint = f"Contexto del turno anterior: {user_turns[-1]}\n"

        prompt_text = QUERY_REWRITE_PROMPT.format(
            question=question,
            context_hint=context_hint,
        )

        try:
            raw = self.rewrite_llm.invoke(prompt_text)
            rewritten = raw.content if hasattr(raw, "content") else str(raw)
            rewritten = rewritten.strip()

            # El modelo a veces envuelve la frase en una explicacion, ej:
            #   "La respuesta es:\nFrase: 'sancion disciplinaria daño bienes'\nTipo de..."
            # Extraemos el contenido util buscando los marcadores del few-shot.
            low_r = rewritten.lower()
            for marker in ("frase:", "consulta:", "busqueda:", "> "):
                if marker in low_r:
                    idx = low_r.rfind(marker)
                    after = rewritten[idx + len(marker):]
                    # Tomar solo la primera linea y limpiar comillas/espacios
                    extracted = after.strip().strip("'\"").split("\n")[0].strip().strip("'\"").strip()
                    if extracted and 4 <= len(extracted) <= _REWRITE_MAX_CHARS:
                        rewritten = extracted
                        break

            # Limpieza final de comillas o espacios sobrantes
            rewritten = rewritten.strip().strip('"').strip("'").strip()

            if _is_bad_rewrite(rewritten, question):
                _logger.debug(
                    "Rewrite descartado — invalido: %r | fallback: %r",
                    rewritten, fallback,
                )
                return fallback

            _logger.debug("Rewrite OK: %r -> %r", question, rewritten)
            return rewritten

        except Exception as exc:
            _logger.warning("Rewrite fallo (%s): fallback '%s'.", exc, fallback)
            return fallback

    def invoke(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Ejecuta la cadena RAG (sin streaming) y retorna respuesta + docs fuente."""
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
        return {"answer": answer, "source_documents": source_docs}

    def _prepare(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]],
    ):
        """
        Nucleo del pipeline RAG. Compartido por invoke e invoke_stream.

        Flujo:
          PASO 1 — Query Rewriting: traduce pregunta coloquial -> terminos normativos
                   para cerrar el Lexical Gap antes de buscar en ChromaDB.
          PASO 2 — Retrieval: busca chunks con la query reescrita.
          PASO 3 — Cobertura: verifica que los chunks recuperados sean relevantes
                   evaluando la UNION de terminos del original + rewrite.
                   (Evita rechazar chunks buenos cuando el original es muy coloquial.)
          PASO 4 — Fallback: si los chunks no cubren la pregunta, busca por keyword.
          PASO 5 — Prompt: construye el texto final para el LLM principal.
        """
        if history is None:
            history = []

        # PASO 1 & 2: rewrite + retrieval
        retrieval_query = self._rewrite_query_for_retrieval(question, history)
        source_docs = self.retriever.invoke(retrieval_query)
        seen_fingerprints: set = set()
        unique_docs = self._filter_and_dedup(source_docs, seen_fingerprints, question)

        # PASO 3 & 4: cobertura con union original+rewrite; fallback keyword si falla
        terms = _key_terms(question)
        needs_fallback = (
            (unique_docs and not _context_covers(
                question, unique_docs, rewritten_query=retrieval_query
            ))
            or (not unique_docs and bool(terms))
        )
        if needs_fallback:
            fallback_docs = self._keyword_search_fallback(terms)
            unique_docs = self._filter_and_dedup(fallback_docs, seen_fingerprints, question)

        # PASO 5: construir prompt
        context = self._format_docs(unique_docs)
        history_text = format_history_for_prompt(history, current_question=question)
        attendance_note = _ATTENDANCE_RULE_NOTE if _is_attendance_question(question) else ""
        
        rights_note = ""
        q_lower = question.lower()
        if "derecho" in q_lower and "deber" not in q_lower and "obligacion" not in q_lower:
            rights_note = (
                "[ADVERTENCIA CRITICA PARA EL LLM]: El usuario está preguntando EXCLUSIVAMENTE por DERECHOS. "
                "Revisa bien los fragmentos: si un fragmento habla de 'acatar', 'cumplir', 'responder', 'asumir', "
                "'tratar a todos...', etc., eso es un DEBER/OBLIGACIÓN. NO LO INCLUYAS EN TU RESPUESTA DE DERECHOS. "
                "Si la información es solo sobre deberes, responde 'No encontré información sobre derechos'.\n"
            )

        prompt_text = self.prompt.format(
            context=context,
            question=question,
            attendance_note=attendance_note,
            rights_note=rights_note,
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
        """Ejecuta la cadena RAG con streaming de tokens."""
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
    """
    Construye la cadena RAG completa.

    - llm:         LLM principal (temperature=TEMPERATURE, 600 tokens, system prompt)
    - rewrite_llm: LLM mini rewriter (temperature=0, 35 tokens, few-shot prompt)
    - prompt:      Template con XML isolation de historial y fragmentos normativos
    """
    llm = create_llm(model_name, temperature)
    rewrite_llm = create_rewrite_llm(model_name)

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question", "attendance_note", "rights_note"],
    )

    return RAGChain(retriever, llm, prompt, rewrite_llm)


def query_rag(
    chain: RAGChain,
    question: str,
    model_name: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Ejecuta una consulta RAG y retorna el resultado estructurado."""
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
    """Formatea la respuesta con citas de fuentes."""
    answer = result["answer"]
    sources = result.get("sources_info", [])
    if not sources:
        return answer
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
