"use client";

import { StatusResponse } from "../lib/api";

interface Props {
  status: StatusResponse | null;
  loading: boolean;
}

export default function StatusBar({ status, loading }: Props) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <span className="w-2 h-2 rounded-full bg-gray-300 animate-pulse" />
        Verificando sistema...
      </div>
    );
  }

  if (!status) {
    return (
      <div className="flex items-center gap-2 text-sm text-red-600">
        <span className="w-2 h-2 rounded-full bg-red-500" />
        Sin conexión con el backend
      </div>
    );
  }

  return (
    <div className="flex items-center gap-4 text-sm flex-wrap">
      <div className="flex items-center gap-1.5">
        <span
          className={`w-2 h-2 rounded-full ${
            status.ollama_running ? "bg-green-500" : "bg-red-500"
          }`}
        />
        <span className={status.ollama_running ? "text-green-700" : "text-red-600"}>
          Ollama {status.ollama_running ? "activo" : "inactivo"}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span
          className={`w-2 h-2 rounded-full ${
            status.vector_store_ready ? "bg-green-500" : "bg-yellow-500"
          }`}
        />
        <span className={status.vector_store_ready ? "text-green-700" : "text-yellow-700"}>
          {status.vector_store_ready ? "Base de conocimiento lista" : "Cargando base..."}
        </span>
      </div>
      {status.active_model && (
        <div className="text-gray-500">
          Modelo: <span className="font-mono font-medium text-gray-700">{status.active_model}</span>
        </div>
      )}
    </div>
  );
}
