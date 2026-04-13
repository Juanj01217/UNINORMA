#!/bin/bash
set -e

echo "======================================="
echo "  UNINORMA Backend - Iniciando..."
echo "======================================="

# --- 1. Esperar a Ollama ---
echo ""
echo "[1/3] Esperando que Ollama esté listo en ${OLLAMA_BASE_URL}..."
until curl -sf "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; do
    echo "  Ollama no disponible aún, reintentando en 3s..."
    sleep 3
done
echo "[1/3] Ollama listo."

# --- 2. Descargar modelo si no está disponible ---
echo ""
echo "[2/3] Verificando modelo qwen2.5:3b..."
python3 - <<'PYEOF'
import requests, os, json, sys

base = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
model = "qwen2.5:3b"

try:
    tags = requests.get(f"{base}/api/tags", timeout=10).json()
    names = [m["name"] for m in tags.get("models", [])]

    if any(model in n for n in names):
        print(f"  Modelo {model} ya disponible.")
        sys.exit(0)

    print(f"  Descargando {model} (~2 GB). Puede tardar 5-10 min la primera vez...")
    with requests.post(f"{base}/api/pull", json={"name": model}, stream=True, timeout=600) as r:
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    completed = data.get("completed", 0)
                    total = data.get("total", 0)
                    if total > 0:
                        pct = int(completed / total * 100)
                        print(f"  {status}: {pct}%", end="\r", flush=True)
                    elif status:
                        print(f"  {status}", flush=True)
                except Exception:
                    pass
    print(f"\n  Modelo {model} descargado correctamente.")

except Exception as e:
    print(f"  ERROR al verificar/descargar modelo: {e}")
    print("  Continuando de todas formas...")
PYEOF
echo "[2/3] Verificación de modelo completada."

# --- 3. Verificar ChromaDB ---
echo ""
echo "[3/3] Verificando base de datos vectorial (ChromaDB)..."
if [ ! -d "/app/data/chroma_db" ] || [ -z "$(ls -A /app/data/chroma_db 2>/dev/null)" ]; then
    echo "  ChromaDB no encontrada. Ejecutando ingestión de datos..."
    python3 ingest.py --pdf-dir /reglamentos
    echo "  Ingestión completada."
else
    echo "[3/3] ChromaDB encontrada y lista."
fi

# --- Iniciar servidor ---
echo ""
echo "======================================="
echo "  API disponible en http://0.0.0.0:8000"
echo "======================================="
exec uvicorn api:app --host 0.0.0.0 --port 8000
