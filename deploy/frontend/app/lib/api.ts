const API_URL = process.env.NEXT_PUBLIC_API_URL || "/api";

// ---------------------------------------------------------------------------
// Benchmark types
// ---------------------------------------------------------------------------

export interface BenchmarkResultItem {
  question_id: string;
  model_name: string;
  question: string;
  answer: string;
  category: string;
  difficulty: string;
  latency_seconds: number;
  memory_usage_mb: number;
  retrieval_hit: boolean;
  answer_relevancy: number;
  faithfulness: number;
  hallucination_detected: boolean;
  no_answer_correct: boolean;
}

export interface ModelSummary {
  total_questions: number;
  successful: number;
  avg_latency_seconds: number;
  retrieval_accuracy: number;
  avg_answer_relevancy: number;
  avg_faithfulness: number;
  hallucination_rate: number;
  avg_memory_mb: number;
}

export interface BenchmarkJob {
  job_id: string;
  status: "running" | "done" | "error";
  started_at: string;
  models: string[];
  quick: boolean;
  total_questions: number;
  completed_questions: number;
  current_model: string | null;
  results: BenchmarkResultItem[];
  summary: Record<string, ModelSummary>;
  error: string | null;
}

export interface BenchmarkRun {
  timestamp: string;
  summary: Record<string, ModelSummary>;
}

export interface Source {
  source: string;
  title: string;
  page: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  model: string;
}

export interface StatusResponse {
  ollama_running: boolean;
  available_models: Record<string, boolean>;
  active_model: string | null;
  vector_store_ready: boolean;
}

export interface ModelsResponse {
  models: string[];
  installed: Record<string, boolean>;
  active: string | null;
  default: string;
}

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("No se pudo conectar con el backend");
  return res.json();
}

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetch(`${API_URL}/models`);
  if (!res.ok) throw new Error("Error al obtener modelos");
  return res.json();
}

export async function loadModel(model: string): Promise<void> {
  const res = await fetch(`${API_URL}/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  if (!res.ok) {
    let message = "Error al cargar el modelo";
    try {
      const err = await res.json();
      message = err.detail || message;
    } catch {
      const text = await res.text();
      if (text) message = text;
    }
    throw new Error(message);
  }
}

// ---------------------------------------------------------------------------
// Benchmark API functions
// ---------------------------------------------------------------------------

export async function startBenchmark(
  models: string[],
  quick: boolean
): Promise<{ job_id: string }> {
  const res = await fetch(`${API_URL}/benchmark/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ models, quick }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Error al iniciar benchmark");
  }
  return res.json();
}

export async function getBenchmarkProgress(jobId: string): Promise<BenchmarkJob> {
  const res = await fetch(`${API_URL}/benchmark/progress/${jobId}`);
  if (!res.ok) throw new Error("Error al obtener progreso del benchmark");
  return res.json();
}

export async function getBenchmarkResults(): Promise<{ runs: BenchmarkRun[] }> {
  const res = await fetch(`${API_URL}/benchmark/results`);
  if (!res.ok) throw new Error("Error al obtener resultados guardados");
  return res.json();
}

export async function sendQuery(
  question: string,
  model: string
): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, model }),
  });
  if (!res.ok) {
    let message = "Error al procesar la consulta";
    try {
      const err = await res.json();
      message = err.detail || message;
    } catch {
      const text = await res.text();
      if (text) message = text;
    }
    throw new Error(message);
  }
  return res.json();
}
