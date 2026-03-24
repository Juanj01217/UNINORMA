#!/bin/bash
# Script de configuracion de Ollama y modelos SLM
# Ejecutar una vez antes de usar el prototipo

echo "============================================="
echo "  Setup Ollama - Prototipo RAG Uninorte"
echo "============================================="

# Verificar si Ollama esta instalado
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "Ollama no esta instalado."
    echo ""
    echo "Instrucciones de instalacion:"
    echo "  Windows: Descarga desde https://ollama.com/download"
    echo "  Linux:   curl -fsSL https://ollama.com/install.sh | sh"
    echo "  macOS:   Descarga desde https://ollama.com/download"
    echo ""
    exit 1
fi

echo "Ollama encontrado: $(ollama --version 2>/dev/null || echo 'version desconocida')"
echo ""

# Modelos a descargar
models=(
    "qwen2.5:1.5b"
    "qwen2.5:3b"
    "llama3.2:1b"
    "llama3.2:3b"
    "phi3:mini"
    "mistral:7b"
)

echo "Se descargaran ${#models[@]} modelos."
echo "Tamano aproximado total: ~15 GB"
echo ""
read -p "Continuar? (s/n): " confirm
if [[ "$confirm" != "s" && "$confirm" != "S" ]]; then
    echo "Cancelado."
    exit 0
fi

echo ""

for model in "${models[@]}"; do
    echo "--- Descargando $model ---"
    ollama pull "$model"
    echo ""
done

echo "============================================="
echo "  Modelos instalados:"
echo "============================================="
ollama list
echo ""
echo "Setup completado."
