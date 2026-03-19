FROM python:3.11-slim

WORKDIR /app

# curl para health checks en el entrypoint
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python y pre-descargar modelo de embeddings
COPY Prototipo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copiar el código fuente y datos (incluye data/chroma_db y data/raw)
COPY Prototipo/ .

# Copiar los PDFs originales como fallback si se necesita re-ingestar
COPY WebScraping/reglamentos/ /reglamentos/

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
