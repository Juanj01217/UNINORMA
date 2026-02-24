const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
    const err = await res.json();
    throw new Error(err.detail || "Error al cargar el modelo");
  }
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
    const err = await res.json();
    throw new Error(err.detail || "Error al procesar la consulta");
  }
  return res.json();
}
