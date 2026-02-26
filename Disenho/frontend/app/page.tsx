"use client";

import { useEffect, useRef, useState } from "react";
import ChatMessage, { Message } from "./components/ChatMessage";
import ModelSelector from "./components/ModelSelector";
import StatusBar from "./components/StatusBar";
import {
  fetchStatus,
  fetchModels,
  loadModel,
  sendQuery,
  StatusResponse,
} from "./lib/api";

const EXAMPLE_QUESTIONS = [
  "¿Cuáles son los derechos de los egresados de Uninorte?",
  "¿Qué dice el reglamento sobre propiedad intelectual de los profesores?",
  "¿Cuál es la jornada laboral en el reglamento interno de trabajo?",
  "¿Qué establece la política de derechos humanos de la universidad?",
  "¿Cuáles son las sanciones por faltas graves en el reglamento interno?",
];

let messageIdCounter = 0;
function newId() {
  return `msg-${++messageIdCounter}`;
}

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("qwen2.5:3b");
  const [modelLoading, setModelLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    async function init() {
      try {
        const s = await fetchStatus();
        setStatus(s);
        if (s.active_model) setSelectedModel(s.active_model);
      } catch {
        setStatus(null);
      } finally {
        setStatusLoading(false);
      }
      try {
        const m = await fetchModels();
        setModels(m.models);
        if (m.default) setSelectedModel(m.default);
      } catch {
        setModels(["qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b", "phi3:mini"]);
      }
    }
    init();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleModelChange(model: string) {
    setSelectedModel(model);
    setModelLoading(true);
    try {
      await loadModel(model);
      const s = await fetchStatus();
      setStatus(s);
    } catch (e) {
      console.error(e);
    } finally {
      setModelLoading(false);
    }
  }

  async function handleSend(question?: string) {
    const text = (question ?? input).trim();
    if (!text || sending) return;

    setInput("");
    setSending(true);

    const userMsg: Message = { id: newId(), role: "user", content: text };
    const loadingMsg: Message = {
      id: newId(),
      role: "assistant",
      content: "",
      loading: true,
    };

    setMessages((prev) => [...prev, userMsg, loadingMsg]);

    try {
      const result = await sendQuery(text, selectedModel);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? { ...m, content: result.answer, sources: result.sources, loading: false }
            : m
        )
      );
      const s = await fetchStatus();
      setStatus(s);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Error desconocido";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? { ...m, content: `Error: ${msg}`, loading: false }
            : m
        )
      );
    } finally {
      setSending(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 shrink-0">
        <div className="max-w-4xl mx-auto flex flex-col gap-3">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-xl font-bold text-gray-900">
                Asistente de Normatividad
              </h1>
              <p className="text-sm text-gray-500 mt-0.5">
                Universidad del Norte · Consulta en lenguaje natural
              </p>
            </div>
            <ModelSelector
              models={models}
              selected={selectedModel}
              onSelect={handleModelChange}
              loading={modelLoading}
            />
          </div>
          <StatusBar status={status} loading={statusLoading} />
        </div>
      </header>

      {/* Chat area */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full min-h-100 gap-6">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h2 className="text-lg font-semibold text-gray-800">
                  ¿Tienes alguna consulta sobre normatividad?
                </h2>
                <p className="text-gray-500 text-sm mt-1 max-w-sm">
                  Pregunta en lenguaje natural. Las respuestas se generan
                  exclusivamente a partir de los documentos oficiales de Uninorte.
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-2xl">
                {EXAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSend(q)}
                    className="text-left px-4 py-3 rounded-xl border border-gray-200 bg-white text-sm text-gray-700 hover:border-blue-400 hover:bg-blue-50 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m) => <ChatMessage key={m.id} message={m} />)
          )}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* Input */}
      <footer className="bg-white border-t border-gray-200 px-4 py-4 shrink-0">
        <div className="max-w-4xl mx-auto flex gap-3 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={sending}
            placeholder="Escribe tu pregunta sobre la normatividad de Uninorte..."
            rows={1}
            className="flex-1 resize-none border border-gray-300 rounded-2xl px-4 py-3 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 max-h-32 overflow-y-auto"
            style={{ minHeight: "48px" }}
          />
          <button
            onClick={() => handleSend()}
            disabled={sending || !input.trim()}
            className="shrink-0 w-12 h-12 rounded-2xl bg-blue-600 text-white flex items-center justify-center hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {sending ? (
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </div>
        <p className="text-center text-xs text-gray-400 mt-2">
          Enter para enviar · Shift+Enter para nueva línea · Prototipo SLM — Uninorte 2025
        </p>
      </footer>
    </div>
  );
}
