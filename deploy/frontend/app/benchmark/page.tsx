"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  BenchmarkJob,
  BenchmarkResultItem,
  BenchmarkRun,
  ModelSummary,
  fetchModels,
  getBenchmarkProgress,
  getBenchmarkResults,
  startBenchmark,
} from "../lib/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const METRIC_LABELS: Record<string, { label: string; unit: string; higherIsBetter: boolean }> = {
  avg_latency_seconds: { label: "Latencia promedio", unit: "s", higherIsBetter: false },
  retrieval_accuracy: { label: "Precisión de recuperación", unit: "%", higherIsBetter: true },
  avg_answer_relevancy: { label: "Relevancia de respuesta", unit: "", higherIsBetter: true },
  avg_faithfulness: { label: "Fidelidad al contexto", unit: "", higherIsBetter: true },
  hallucination_rate: { label: "Tasa de alucinación", unit: "%", higherIsBetter: false },
  avg_memory_mb: { label: "Memoria promedio", unit: "MB", higherIsBetter: false },
};

function formatMetric(key: string, value: number): string {
  const cfg = METRIC_LABELS[key];
  if (!cfg) return String(value);
  if (cfg.unit === "%" ) return `${(value * 100).toFixed(1)}%`;
  if (cfg.unit === "s") return `${value.toFixed(2)}s`;
  if (cfg.unit === "MB") return `${value.toFixed(1)} MB`;
  return value.toFixed(3);
}

function getBestModel(
  summary: Record<string, ModelSummary>,
  key: keyof ModelSummary
): string | null {
  const models = Object.keys(summary);
  if (models.length < 2) return null;
  const cfg = METRIC_LABELS[key];
  if (!cfg) return null;
  return models.reduce((best, m) => {
    const bv = summary[best][key] as number;
    const mv = summary[m][key] as number;
    return cfg.higherIsBetter ? (mv > bv ? m : best) : (mv < bv ? m : best);
  });
}

function progressPercent(job: BenchmarkJob): number {
  if (!job.total_questions) return 0;
  return Math.round((job.completed_questions / job.total_questions) * 100);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetricCard({
  metricKey,
  summary,
}: {
  metricKey: string;
  summary: Record<string, ModelSummary>;
}) {
  const cfg = METRIC_LABELS[metricKey];
  const bestModel = getBestModel(summary, metricKey as keyof ModelSummary);
  const models = Object.keys(summary);

  const values = models.map((m) => ({
    model: m,
    value: summary[m][metricKey as keyof ModelSummary] as number,
  }));

  const maxVal = Math.max(...values.map((v) => v.value));
  const minVal = Math.min(...values.map((v) => v.value));
  const range = maxVal - minVal || 1;

  return (
    <div className="bg-white border border-gray-200 rounded-2xl p-5 flex flex-col gap-3">
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-semibold text-gray-700">{cfg.label}</span>
        <span className="text-xs text-gray-400">
          {cfg.higherIsBetter ? "↑ mayor es mejor" : "↓ menor es mejor"}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {values.map(({ model, value }) => {
          const isBest = model === bestModel;
          const barWidth =
            cfg.higherIsBetter
              ? ((value - minVal) / range) * 100
              : ((maxVal - value) / range) * 100;

          return (
            <div key={model} className="flex flex-col gap-1">
              <div className="flex items-center justify-between text-xs">
                <span className={`font-medium ${isBest ? "text-green-700" : "text-gray-600"}`}>
                  {model}
                  {isBest && models.length > 1 && (
                    <span className="ml-1 text-green-600">✓ mejor</span>
                  )}
                </span>
                <span className={`font-mono ${isBest ? "text-green-700 font-semibold" : "text-gray-500"}`}>
                  {formatMetric(metricKey, value)}
                </span>
              </div>
              <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${isBest ? "bg-green-500" : "bg-blue-400"}`}
                  style={{ width: `${Math.max(4, barWidth)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SummaryTable({ summary }: { summary: Record<string, ModelSummary> }) {
  const models = Object.keys(summary);
  const metrics = Object.keys(METRIC_LABELS) as (keyof ModelSummary)[];

  return (
    <div className="overflow-x-auto rounded-2xl border border-gray-200">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-50 border-b border-gray-200">
            <th className="text-left px-4 py-3 font-semibold text-gray-600 whitespace-nowrap">
              Métrica
            </th>
            {models.map((m) => (
              <th key={m} className="text-center px-4 py-3 font-semibold text-gray-800 whitespace-nowrap">
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((key, idx) => {
            const cfg = METRIC_LABELS[key];
            const bestModel = getBestModel(summary, key);
            return (
              <tr key={key} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50/50"}>
                <td className="px-4 py-2.5 text-gray-600 font-medium whitespace-nowrap">
                  {cfg.label}
                  <span className="ml-1 text-gray-400 text-xs">
                    {cfg.higherIsBetter ? "↑" : "↓"}
                  </span>
                </td>
                {models.map((m) => {
                  const val = summary[m][key] as number;
                  const isBest = m === bestModel && models.length > 1;
                  return (
                    <td
                      key={m}
                      className={`text-center px-4 py-2.5 font-mono font-medium ${
                        isBest ? "text-green-700 bg-green-50" : "text-gray-700"
                      }`}
                    >
                      {formatMetric(key, val)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
          <tr className="bg-gray-50 border-t border-gray-200">
            <td className="px-4 py-2.5 text-gray-600 font-medium">Preguntas evaluadas</td>
            {models.map((m) => (
              <td key={m} className="text-center px-4 py-2.5 font-mono text-gray-700">
                {summary[m].successful}/{summary[m].total_questions}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function ResultsDetail({ results }: { results: BenchmarkResultItem[] }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border border-gray-200 rounded-2xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-3.5 bg-gray-50 hover:bg-gray-100 transition-colors text-sm font-semibold text-gray-700"
      >
        <span>Respuestas por pregunta ({results.length} items)</span>
        <svg
          className={`w-4 h-4 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="divide-y divide-gray-100 max-h-[600px] overflow-y-auto">
          {results.map((r, i) => (
            <div key={i} className="px-5 py-4 text-sm">
              <div className="flex flex-wrap gap-2 mb-2">
                <span className="px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 text-xs font-medium">
                  {r.model_name}
                </span>
                <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 text-xs">
                  {r.category}
                </span>
                <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 text-xs">
                  {r.difficulty}
                </span>
                <span
                  className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                    r.retrieval_hit
                      ? "bg-green-100 text-green-700"
                      : "bg-red-100 text-red-700"
                  }`}
                >
                  retrieval {r.retrieval_hit ? "✓" : "✗"}
                </span>
                <span className="ml-auto font-mono text-gray-500 text-xs">
                  {r.latency_seconds >= 0 ? `${r.latency_seconds.toFixed(2)}s` : "ERROR"}
                </span>
              </div>

              <p className="font-medium text-gray-800 mb-1">{r.question}</p>
              <p className="text-gray-600 text-xs leading-relaxed line-clamp-3">{r.answer}</p>

              <div className="flex gap-4 mt-2 text-xs text-gray-500">
                <span>Relevancia: <strong>{r.answer_relevancy.toFixed(2)}</strong></span>
                <span>Fidelidad: <strong>{r.faithfulness.toFixed(2)}</strong></span>
                {r.hallucination_detected && (
                  <span className="text-orange-600 font-medium">⚠ alucinación detectada</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PastRuns({ runs }: { runs: BenchmarkRun[] }) {
  if (!runs.length) return null;

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-base font-semibold text-gray-700">Ejecuciones anteriores</h2>
      {runs.map((run) => (
        <div key={run.timestamp} className="border border-gray-200 rounded-2xl p-5">
          <p className="text-xs text-gray-400 mb-3 font-mono">{run.timestamp}</p>
          <SummaryTable summary={run.summary} />
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function BenchmarkPage() {
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [quickMode, setQuickMode] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<BenchmarkJob | null>(null);
  const [pastRuns, setPastRuns] = useState<BenchmarkRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load models and past results on mount
  useEffect(() => {
    fetchModels()
      .then((m) => {
        setAvailableModels(m.models);
        setSelectedModels(m.models.slice(0, 2)); // pre-select first 2
      })
      .catch(() => {
        const fallback = ["qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b", "phi3:mini"];
        setAvailableModels(fallback);
        setSelectedModels(fallback.slice(0, 2));
      });

    getBenchmarkResults()
      .then((r) => setPastRuns(r.runs))
      .catch(() => {});
  }, []);

  // Polling logic
  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollJob = useCallback(
    (id: string) => {
      pollRef.current = setInterval(async () => {
        try {
          const j = await getBenchmarkProgress(id);
          setJob(j);
          if (j.status === "done" || j.status === "error") {
            stopPolling();
            // Refresh past runs
            getBenchmarkResults()
              .then((r) => setPastRuns(r.runs))
              .catch(() => {});
          }
        } catch {
          stopPolling();
        }
      }, 2000);
    },
    [stopPolling]
  );

  useEffect(() => () => stopPolling(), [stopPolling]);

  async function handleStart() {
    if (!selectedModels.length) {
      setError("Selecciona al menos un modelo.");
      return;
    }
    setError(null);
    setStarting(true);
    setJob(null);
    try {
      const { job_id } = await startBenchmark(selectedModels, quickMode);
      setJobId(job_id);
      // Fetch initial state
      const initial = await getBenchmarkProgress(job_id);
      setJob(initial);
      pollJob(job_id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Error desconocido");
    } finally {
      setStarting(false);
    }
  }

  function toggleModel(m: string) {
    setSelectedModels((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]
    );
  }

  const isRunning = job?.status === "running";
  const isDone = job?.status === "done";
  const isError = job?.status === "error";
  const pct = job ? progressPercent(job) : 0;

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 shrink-0">
        <div className="max-w-5xl mx-auto flex items-center gap-4">
          <Link
            href="/"
            className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-800 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Chat
          </Link>
          <div className="h-4 w-px bg-gray-200" />
          <div>
            <h1 className="text-xl font-bold text-gray-900">Benchmark de modelos</h1>
            <p className="text-sm text-gray-500">
              Compara latencia, calidad y uso de recursos entre modelos SLM
            </p>
          </div>
        </div>
      </header>

      <main className="flex-1 px-4 py-8">
        <div className="max-w-5xl mx-auto flex flex-col gap-8">

          {/* Configuration */}
          <div className="bg-white border border-gray-200 rounded-2xl p-6 flex flex-col gap-5">
            <h2 className="font-semibold text-gray-800">Configuración del benchmark</h2>

            {/* Model checkboxes */}
            <div>
              <p className="text-sm text-gray-600 mb-3">
                Selecciona los modelos a comparar:
              </p>
              <div className="flex flex-wrap gap-2">
                {availableModels.map((m) => {
                  const checked = selectedModels.includes(m);
                  return (
                    <button
                      key={m}
                      onClick={() => toggleModel(m)}
                      disabled={isRunning}
                      className={`px-4 py-2 rounded-xl border text-sm font-medium transition-colors ${
                        checked
                          ? "border-blue-500 bg-blue-50 text-blue-700"
                          : "border-gray-200 bg-white text-gray-600 hover:border-gray-300"
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {m}
                    </button>
                  );
                })}
              </div>
              {selectedModels.length > 0 && (
                <p className="text-xs text-gray-400 mt-2">
                  {selectedModels.length} modelo{selectedModels.length > 1 ? "s" : ""} seleccionado{selectedModels.length > 1 ? "s" : ""}
                </p>
              )}
            </div>

            {/* Mode toggle */}
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-600 font-medium">Modo:</span>
              <div className="flex rounded-xl border border-gray-200 overflow-hidden text-sm">
                <button
                  onClick={() => setQuickMode(true)}
                  disabled={isRunning}
                  className={`px-4 py-2 transition-colors ${
                    quickMode ? "bg-blue-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
                  } disabled:opacity-50`}
                >
                  Rápido (6 preguntas)
                </button>
                <button
                  onClick={() => setQuickMode(false)}
                  disabled={isRunning}
                  className={`px-4 py-2 transition-colors border-l border-gray-200 ${
                    !quickMode ? "bg-blue-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
                  } disabled:opacity-50`}
                >
                  Completo (~40 preguntas)
                </button>
              </div>
              <span className="text-xs text-gray-400">
                {quickMode
                  ? "~1–3 min por modelo"
                  : "~10–30 min por modelo"}
              </span>
            </div>

            {error && (
              <p className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-2">
                {error}
              </p>
            )}

            <button
              onClick={handleStart}
              disabled={isRunning || starting || !selectedModels.length}
              className="self-start px-6 py-2.5 rounded-xl bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {isRunning || starting ? (
                <span className="flex items-center gap-2">
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                  </svg>
                  Ejecutando…
                </span>
              ) : (
                "Iniciar benchmark"
              )}
            </button>
          </div>

          {/* Progress */}
          {job && (isRunning || isDone || isError) && (
            <div className="bg-white border border-gray-200 rounded-2xl p-6 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-gray-800">
                  {isRunning ? "En progreso…" : isDone ? "Benchmark completado" : "Error"}
                </h2>
                <span
                  className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                    isRunning
                      ? "bg-blue-100 text-blue-700"
                      : isDone
                      ? "bg-green-100 text-green-700"
                      : "bg-red-100 text-red-700"
                  }`}
                >
                  {isRunning ? "corriendo" : isDone ? "listo" : "error"}
                </span>
              </div>

              {isError && (
                <p className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-2">
                  {job.error}
                </p>
              )}

              {(isRunning || isDone) && (
                <>
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>
                        {isRunning && job.current_model
                          ? `Procesando: ${job.current_model}`
                          : `${job.completed_questions} / ${job.total_questions} preguntas`}
                      </span>
                      <span>{pct}%</span>
                    </div>
                    <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full transition-all duration-500"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="text-gray-400">Modelos:</span>
                    {job.models.map((m) => (
                      <span
                        key={m}
                        className={`px-2 py-0.5 rounded-full font-medium ${
                          job.current_model === m
                            ? "bg-blue-100 text-blue-700"
                            : job.summary[m]
                            ? "bg-green-100 text-green-700"
                            : "bg-gray-100 text-gray-500"
                        }`}
                      >
                        {job.current_model === m && (
                          <span className="inline-block w-1.5 h-1.5 bg-blue-500 rounded-full mr-1 animate-pulse" />
                        )}
                        {m}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Live / final results */}
          {job && Object.keys(job.summary).length > 0 && (
            <div className="flex flex-col gap-6">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-gray-800 text-base">
                  Resultados comparativos
                  {isRunning && (
                    <span className="ml-2 text-xs text-blue-500 font-normal">actualizando…</span>
                  )}
                </h2>
                <span className="text-xs text-gray-400 font-mono">
                  job: {jobId}
                </span>
              </div>

              {/* Metric cards grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.keys(METRIC_LABELS).map((key) => (
                  <MetricCard key={key} metricKey={key} summary={job.summary} />
                ))}
              </div>

              {/* Comparison table */}
              <SummaryTable summary={job.summary} />

              {/* Per-question detail */}
              {isDone && job.results.length > 0 && (
                <ResultsDetail results={job.results} />
              )}
            </div>
          )}

          {/* Past runs */}
          {!job && <PastRuns runs={pastRuns} />}
        </div>
      </main>
    </div>
  );
}
