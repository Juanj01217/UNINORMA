# Prototipo RAG - Asistente de Normatividad Uninorte

Prototipo de asistente virtual basado en Small Language Models (SLM) y arquitectura RAG para consultar la normatividad institucional de la Universidad del Norte en lenguaje natural.

## Arquitectura

```
                          INGESTA (offline)
  ┌─────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
  │  PDFs   │───>│  Extraccion  │───>│ Chunking │───>│Embeddings│
  │(10 docs)│    │  (PyMuPDF)   │    │(1000 chr)│    │(MiniLM)  │
  └─────────┘    └──────────────┘    └──────────┘    └────┬─────┘
                                                          │
                                                     ┌────▼─────┐
                                                     │ ChromaDB │
                                                     │(vectores)│
                                                     └────┬─────┘
                          CONSULTA (online)                │
  ┌─────────┐    ┌──────────────┐    ┌──────────┐    ┌───▼──────┐
  │ Usuario │───>│   Pregunta   │───>│Similarity│───>│ Top-K    │
  │  (chat) │    │  (espanol)   │    │ Search   │    │ Chunks   │
  └─────────┘    └──────────────┘    └──────────┘    └────┬─────┘
       ▲                                                   │
       │         ┌──────────────┐    ┌──────────┐    ┌────▼─────┐
       └─────────│  Respuesta   │<───│  Ollama  │<───│  Prompt  │
                 │ + Fuentes    │    │   SLM    │    │(contexto)│
                 └──────────────┘    └──────────┘    └──────────┘
```

## Stack Tecnologico

| Componente | Tecnologia | Costo |
|---|---|---|
| Inferencia SLM | Ollama | Gratuito |
| Embeddings | sentence-transformers (MiniLM) | Gratuito |
| Vector Store | ChromaDB | Gratuito |
| Extraccion PDF | PyMuPDF | Gratuito |
| Orquestacion | LangChain | Gratuito |
| Interfaz | Gradio | Gratuito |

## Modelos SLM a Evaluar

| Modelo | Parametros | RAM aprox. |
|---|---|---|
| qwen2.5:1.5b | 1.5B | ~2 GB |
| qwen2.5:3b | 3B | ~3 GB |
| llama3.2:1b | 1B | ~1.5 GB |
| llama3.2:3b | 3B | ~3 GB |
| phi3:mini | 3.8B | ~4 GB |
| mistral:7b | 7B | ~8 GB |

## Requisitos

- Python 3.10+
- Ollama instalado ([descargar](https://ollama.com/download))
- 8 GB RAM minimo (16 GB recomendado para modelos 7B)
- ~15 GB disco para todos los modelos

## Instalacion

### 1. Entorno virtual

```bash
cd Disenho/Prototipo
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 2. Dependencias Python

```bash
pip install -r requirements.txt
```

### 3. Instalar Ollama y modelos

```bash
# Instalar Ollama (ver https://ollama.com/download)

# Descargar todos los modelos
bash setup_ollama.sh

# O descargar uno solo para probar rapido:
ollama pull qwen2.5:3b
```

### 4. Preparar datos (si no se ha hecho)

Ejecutar el notebook de WebScraping para descargar los PDFs:

```bash
cd ../WebScraping
jupyter notebook dataCollection.ipynb  # Ejecutar todas las celdas
cd ../Prototipo
```

### 5. Ejecutar ingesta

```bash
python ingest.py
```

Esto extrae texto de los PDFs, lo divide en chunks, genera embeddings y los almacena en ChromaDB.

## Uso

### Interfaz Web (Gradio)

```bash
python app.py
# Abre http://localhost:7860
```

### CLI

```bash
# Consulta unica
python query.py "Cuales son los derechos de los egresados?"

# Modo interactivo
python query.py --interactive

# Con modelo especifico
python query.py --model phi3:mini "Que dice el reglamento de profesores?"
```

### Benchmark de Modelos

```bash
# Evaluar todos los modelos instalados
python -m benchmark.run_benchmark

# Evaluar modelos especificos
python -m benchmark.run_benchmark --models qwen2.5:3b phi3:mini

# Visualizar resultados
jupyter notebook benchmark/analysis.ipynb
```

## Estructura del Proyecto

```
Prototipo/
├── config.py              # Configuracion centralizada
├── ingest.py              # Pipeline de ingesta PDF -> ChromaDB
├── query.py               # CLI de consultas
├── app.py                 # Interfaz web Gradio
├── setup_ollama.sh        # Script de instalacion de modelos
├── requirements.txt       # Dependencias Python
├── src/
│   ├── pdf_extractor.py   # Extraccion de texto de PDFs
│   ├── text_chunker.py    # Division en chunks
│   ├── embeddings.py      # Modelo de embeddings
│   ├── vector_store.py    # Gestion de ChromaDB
│   ├── rag_chain.py       # Cadena RAG (retriever + LLM)
│   ├── ollama_client.py   # Cliente Ollama
│   └── prompt_templates.py # Prompts en espanol
├── benchmark/
│   ├── test_questions.json # 25 preguntas de evaluacion
│   ├── run_benchmark.py   # Ejecutor de benchmarks
│   ├── metrics.py         # Metricas de evaluacion
│   ├── analysis.ipynb     # Notebook de visualizacion
│   └── results/           # Resultados (gitignored)
└── data/
    ├── raw/               # PDFs originales (gitignored)
    ├── processed/         # Texto extraido (gitignored)
    └── chroma_db/         # Vector store (gitignored)
```

## Verificar Estado del Sistema

```bash
python -c "from src.ollama_client import print_status; print_status()"
```
