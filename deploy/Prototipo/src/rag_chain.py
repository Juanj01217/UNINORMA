"""Cadena RAG: combina retriever + LLM para Q&A sobre normatividad."""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, DEFAULT_SLM_MODEL, TEMPERATURE, MAX_TOKENS
from src.prompt_templates import (
    SYSTEM_PROMPT_ES,
    RAG_PROMPT_TEMPLATE,
    format_context_from_docs,
    format_history_for_prompt,
    build_retrieval_query,
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
        if history is None:
            history = []

        # 1. Query enriquecida para recuperacion contextual
        retrieval_query = build_retrieval_query(question, history)

        # 2. Recuperar documentos
        source_docs = self.retriever.invoke(retrieval_query)
        context = self._format_docs(source_docs)

        # 3. Formatear historial para el prompt
        history_text = format_history_for_prompt(history)

        # 4. Construir prompt y llamar al LLM directamente
        prompt_text = self.prompt.format(
            context=context,
            question=question,
            history=history_text,
        )
        raw_answer = self.llm.invoke(prompt_text)

        # Normalizar salida (string o AIMessage)
        answer = raw_answer.content if hasattr(raw_answer, "content") else str(raw_answer)

        return {
            "answer": answer,
            "source_documents": source_docs,
        }


def create_rag_chain(
    retriever,
    model_name: str = DEFAULT_SLM_MODEL,
    temperature: float = TEMPERATURE,
) -> RAGChain:
    """
    Construye la cadena RAG completa: retriever -> prompt -> LLM -> respuesta.
    """
    llm = create_llm(model_name, temperature)

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question", "history"],
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
