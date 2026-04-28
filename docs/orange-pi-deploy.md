# Despliegue en Orange Pi 5 Pro / Plus (Rockchip RK3588)

Esta guia complementa el `README.md` principal y detalla los pasos para
correr UNINORMA en un Orange Pi 5 Pro / Plus aprovechando la NPU de 6 TOPS
del SoC RK3588.

> **Variables de entorno de las que depende todo:** `LLM_BACKEND`,
> `EMBEDDER_BACKEND`, `RERANKER_ENABLED`, `RKLLM_MODEL_PATH`. Todas tienen
> defaults sensatos en `deploy/Prototipo/config.py`.

## 1. Preparar el host

1. SO recomendado: **Ubuntu 22.04 / 24.04 ARM64** con kernel >= 5.10 con
   soporte oficial de Rockchip (imagenes de Joshua Riek o de Armbian con
   parche Rockchip).
2. Verificar que el dispositivo NPU este expuesto:
   ```bash
   ls -l /dev/dri /dev/rga
   ```
   Debe haber al menos `/dev/dri/card0`. Si no, falta el driver `rknpu`.
3. Instalar Docker + Compose plugin (`docker compose version` >= 2.20).
4. Anadir el usuario a los grupos `video` y `render`:
   ```bash
   sudo usermod -aG video,render $USER && newgrp video
   ```

## 2. Pre-generar la base vectorial fuera del Pi

La ingestion (scraping + extraccion + chunkeo + embeddings) tarda varios
minutos y no debe correr en el Pi. Hazla **una vez** en tu maquina de
desarrollo:

```bash
docker compose build backend
docker compose run --rm backend python ingest.py --pdf-dir /reglamentos
```

Esto deja el volumen `chroma_data` poblado. Para llevarlo al Pi:

```bash
# en el equipo de desarrollo
docker run --rm -v chroma_data:/data -v $(pwd):/backup alpine \
    tar czf /backup/chroma_data.tar.gz -C /data .

# copiar el tarball al Pi y restaurarlo en el volumen del Pi
scp chroma_data.tar.gz orangepi:~/
ssh orangepi
docker volume create chroma_data
docker run --rm -v chroma_data:/data -v ~:/backup alpine \
    tar xzf /backup/chroma_data.tar.gz -C /data
```

A partir de aqui el `entrypoint.sh` detecta la BD existente y **no**
re-ingesta (ver `deploy/Prototipo/entrypoint.sh:61`).

## 3. Convertir el SLM a `.rkllm`

El runtime RKLLM no acepta GGUF; necesita su formato propio.

```bash
# en una maquina x86 con CUDA (rkllm-toolkit usa GPU para cuantizar)
pip install rkllm-toolkit
python -m rkllm.convert \
    --model qwen2.5-1.5b-instruct \
    --target rk3588 \
    --quant w8a8 \
    --output qwen2.5-1.5b-instruct.rkllm
```

Tambien hay modelos ya convertidos en HuggingFace
(`huggingface.co/<usuario>/...rkllm`). Coloca el `.rkllm` resultante en
`deploy/models/qwen2.5-1.5b-instruct.rkllm` (gitignored).

## 4. Levantar el stack en el Pi

Usa el override `docker-compose.orangepi.yml` que activa `linux/arm64`,
expone `/dev/dri` al backend y por default desactiva el reranker pesado:

```bash
LLM_BACKEND=rkllm \
LLM_MODEL=qwen2.5:1.5b \
EMBEDDER_BACKEND=onnx \
RERANKER_ENABLED=false \
docker compose \
    -f docker-compose.yml \
    -f docker-compose.orangepi.yml \
    up -d --build
```

Si la conversion RKLLM aun no esta lista, deja `LLM_BACKEND=ollama`: el
stack arranca igual y el responder corre en CPU ARM (mas lento, pero
funcional).

## 5. Verificacion

```bash
# 1. La BD vectorial NO se reingesta
docker compose logs backend | grep "ChromaDB"
# -> debe mostrar "ChromaDB encontrada y lista."

# 2. El backend usa el backend correcto
docker compose exec backend python -c "from config import LLM_BACKEND; print(LLM_BACKEND)"

# 3. Latencia E2E
curl -s http://localhost:5174/api/ask \
    -H 'content-type: application/json' \
    -d '{"question":"Cuantas faltas justifican un reporte?"}' \
    -w "\n[%{time_total}s]\n"
```

Objetivo razonable en Orange Pi 5 Pro 8 GB con `LLM_BACKEND=rkllm`:
**< 4 s** end-to-end para preguntas que aciertan en retrieval.

## 6. Troubleshooting

| Sintoma | Probable causa | Fix |
|---|---|---|
| `RKLLM_BACKEND=rkllm pero el runtime no esta disponible` | Falta `librkllmrt.so` o no hay `.rkllm` montado | Revisar `docker compose exec backend ls /app/models` |
| Backend OOM-killed al primer request | Reranker activo + Ollama cargando | `RERANKER_ENABLED=false`, considerar `qwen2.5:1.5b` con `q4_K_M` |
| Cada `up` reingesta los PDFs | Volumen `chroma_data` no creado o vacio | Repetir paso 2; comprobar `docker volume ls` |
| Modelo Ollama distinto al esperado | `LLM_MODEL` no propagado | Asegurar export de la env var antes de `docker compose up` |
