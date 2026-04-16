# Benchmark Comparativo de Modelos SLM — UNINORMA

Documentación técnica del sistema de evaluación automática que compara modelos de lenguaje en términos de calidad de respuesta, velocidad y uso de recursos.

---

## ¿Qué es el benchmark y para qué sirve?

El benchmark es una herramienta de evaluación automática que mide el rendimiento de diferentes modelos de lenguaje (SLM) al responder preguntas sobre la normatividad de Uninorte. Permite responder preguntas como:

- ¿Qué modelo responde más rápido?
- ¿Cuál recupera mejor los documentos correctos?
- ¿Cuál alucina menos?
- ¿Cuál usa menos memoria RAM?
- ¿Cuál produce respuestas más relevantes y fieles al contexto?

---

## Archivos involucrados

| Archivo | Rol |
|---|---|
| `deploy/Prototipo/api.py` | Endpoints REST del benchmark (backend) |
| `deploy/Prototipo/benchmark/metrics.py` | Funciones de cálculo de métricas |
| `deploy/Prototipo/benchmark/run_benchmark.py` | Ejecutor CLI (uso por línea de comandos) |
| `deploy/Prototipo/benchmark/test_questions.json` | Banco de 25 preguntas de evaluación |
| `deploy/Prototipo/benchmark/results/` | Resultados guardados en disco (JSON) |
| `deploy/frontend/app/benchmark/page.tsx` | Página de visualización del benchmark |
| `deploy/frontend/app/lib/api.ts` | Funciones de comunicación frontend ↔ backend |

---

## Arquitectura del sistema de benchmark

```
NAVEGADOR (Next.js)
│
├── Usuario selecciona modelos y hace clic en "Iniciar"
│
└─> POST /benchmark/start
    │   { models: ["qwen2.5:3b", "qwen2.5:1.5b"], quick: true }
    │
    └─> Backend crea un job_id y lanza un hilo en segundo plano
        │
        │   Mientras corre el hilo:
        │   ┌─────────────────────────────────────────┐
        │   │  Para cada modelo:                       │
        │   │    Para cada pregunta de prueba:         │
        │   │      1. Corre el pipeline RAG completo   │
        │   │      2. Mide latencia, memoria           │
        │   │      3. Calcula 5 métricas de calidad    │
        │   │      4. Actualiza el estado del job      │
        │   └─────────────────────────────────────────┘
        │
        Frontend hace polling cada 2 segundos:
        └─> GET /benchmark/progress/{job_id}
            │   { status: "running", completed: 3, total: 12, summary: {...} }
            │
            └─> Actualiza barras de progreso y tarjetas en tiempo real

Al finalizar:
└─> Guarda resultados en benchmark/results/{timestamp}_summary.json
└─> GET /benchmark/results  ← muestra ejecuciones anteriores
```

---

## API REST del benchmark

### `POST /benchmark/start`

Inicia un nuevo job de benchmark en un hilo separado (no bloquea el servidor).

**Request:**
```json
{
  "models": ["qwen2.5:3b", "qwen2.5:1.5b"],
  "quick": true
}
```

**Response:**
```json
{
  "job_id": "a3f9c1b2"
}
```

- `models`: lista de modelos a comparar (deben estar instalados en Ollama)
- `quick: true`: usa 6 preguntas (1 por categoría) — ~1-3 min por modelo
- `quick: false`: usa las 25 preguntas completas — ~10-30 min por modelo

---

### `GET /benchmark/progress/{job_id}`

Retorna el estado actual del job para hacer polling desde el frontend.

**Response:**
```json
{
  "job_id": "a3f9c1b2",
  "status": "running",
  "started_at": "2025-04-16T14:30:00",
  "models": ["qwen2.5:3b", "qwen2.5:1.5b"],
  "quick": true,
  "total_questions": 12,
  "completed_questions": 5,
  "current_model": "qwen2.5:3b",
  "results": [...],
  "summary": {
    "qwen2.5:3b": {
      "avg_latency_seconds": 8.42,
      "retrieval_accuracy": 0.833,
      "avg_answer_relevancy": 0.712,
      "avg_faithfulness": 0.885,
      "hallucination_rate": 0.0,
      "avg_memory_mb": 12.4
    }
  },
  "error": null
}
```

El `status` puede ser `"running"`, `"done"` o `"error"`.

---

### `GET /benchmark/results`

Retorna hasta 10 ejecuciones anteriores guardadas en disco.

**Response:**
```json
{
  "runs": [
    {
      "timestamp": "20250416_143022",
      "summary": { "qwen2.5:3b": {...}, "qwen2.5:1.5b": {...} }
    }
  ]
}
```

---

## Preguntas de prueba (`test_questions.json`)

El benchmark usa 25 preguntas diseñadas para evaluar distintos aspectos del sistema RAG. Cada pregunta tiene:

| Campo | Descripción |
|---|---|
| `id` | Identificador único (q001–q025) |
| `question` | Pregunta en lenguaje natural |
| `expected_source` | Nombre del PDF donde debe estar la respuesta |
| `category` | Categoría temática |
| `difficulty` | Nivel de dificultad |

### Categorías de preguntas

| Categoría | Descripción | Ejemplo |
|---|---|---|
| `reglamento_estudiante` | Reglamento estudiantil | Notas mínimas para aprobar |
| `reglamento_profesor` | Reglamento de profesores | Requisitos de contratación |
| `reglamento_trabajo` | Reglamento interno de trabajo | Jornada laboral, sanciones |
| `reglamento_egresados` | Reglamento de egresados | Derechos y deberes |
| `propiedad_intelectual` | Reglamento de propiedad intelectual | Obras de profesores |
| `derechos_humanos` | Política de derechos humanos | Mecanismos de resolución |
| `bienestar` | Servicios de bienestar | Servicios universitarios |
| `protocolo` | Protocolo institucional | Protocolo ante acoso |
| `resoluciones` | Resoluciones específicas | Resolución del Consejo Académico |
| `cross_reference` | Preguntas que cruzan dos documentos | Relación entre reglamentos |
| `negation` | Preguntas sin respuesta en los documentos | Teletrabajo, salarios, rector |

### Niveles de dificultad

- **easy**: La respuesta está directamente en un documento, bien indexada.
- **medium**: Requiere síntesis de varios párrafos o comparación dentro del documento.
- **hard**: Cruza múltiples documentos (`cross_reference`) o la respuesta no existe en los documentos (`negation`).

Las preguntas `negation` son especialmente importantes: evalúan si el modelo sabe decir "no tengo información" en lugar de inventar una respuesta.

---

## Métricas explicadas

### 1. Latencia promedio (`avg_latency_seconds`)
**Qué mide:** Tiempo total en segundos desde que se envía la pregunta hasta recibir la respuesta completa del modelo.

**Cómo se calcula:**
```python
start = time.time()
result = query_rag(chain, question, model)
latency = time.time() - start
```

**Interpretación:** Menor es mejor. Incluye búsqueda en ChromaDB + generación de tokens por Ollama. Varía según el hardware y el tamaño del modelo.

---

### 2. Precisión de recuperación (`retrieval_accuracy`)
**Qué mide:** Qué porcentaje de preguntas recuperó el documento correcto (el indicado en `expected_source`).

**Cómo se calcula:**
```python
# Verifica si el nombre del PDF esperado aparece entre los documentos recuperados
for source in retrieved_sources:
    if expected_source in source or source in expected_source:
        return True  # hit
```

**Interpretación:** Valor entre 0 y 1 (o 0% a 100%). Un valor alto significa que ChromaDB está encontrando los documentos relevantes correctamente.

---

### 3. Relevancia de respuesta (`avg_answer_relevancy`)
**Qué mide:** Qué tan relacionada está la respuesta del modelo con la pregunta original.

**Cómo se calcula:** Similitud coseno entre el embedding de la pregunta y el embedding de la respuesta, usando el modelo `paraphrase-multilingual-MiniLM-L12-v2`.

```python
q_emb = sentence_model.encode([question])[0]
a_emb = sentence_model.encode([answer])[0]
cosine_sim = dot(q_emb, a_emb) / (norm(q_emb) * norm(a_emb))
```

**Interpretación:** Valor entre 0 y 1. Un valor alto indica que la respuesta está semánticamente alineada con la pregunta. No mide si es correcta, sino si es pertinente.

---

### 4. Fidelidad al contexto (`avg_faithfulness`)
**Qué mide:** Qué porcentaje de la respuesta del modelo está respaldado por los documentos recuperados (contexto RAG).

**Cómo se calcula:**
```python
# Divide la respuesta en oraciones
# Para cada oración, verifica si sus palabras clave aparecen en el contexto
# Si ≥50% de palabras clave están en el contexto, la oración se considera "anclada"
grounded / total_sentences
```

**Interpretación:** Valor entre 0 y 1. Un valor bajo puede indicar que el modelo está inventando información no presente en los documentos.

---

### 5. Tasa de alucinación (`hallucination_rate`)
**Qué mide:** Porcentaje de respuestas donde el modelo introduce números, fechas o entidades significativas que no aparecen en el contexto recuperado.

**Cómo se calcula:**
```python
answer_numbers = set(re.findall(r'\b\d+\b', answer))
context_numbers = set(re.findall(r'\b\d+\b', context))
new_numbers = answer_numbers - context_numbers
# Si hay más de 2 números nuevos y significativos (>10), se marca como alucinación
```

**Interpretación:** Valor entre 0 y 1 (o 0% a 100%). Menor es mejor. Un modelo que inventa artículos, fechas o cifras que no están en los documentos tendrá una tasa alta.

---

### 6. Memoria promedio (`avg_memory_mb`)
**Qué mide:** Incremento en el uso de RAM del proceso Python durante cada consulta.

**Cómo se calcula:**
```python
mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
result = query_rag(...)
mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
delta = max(0, mem_after - mem_before)
```

**Interpretación:** En MB. Menor es mejor. Nota: la mayor parte de la memoria de los modelos la gestiona Ollama en su propio proceso; este valor mide solo el overhead del pipeline Python (embeddings, LangChain, ChromaDB).

---

## Flujo de ejecución paso a paso

```
1. Usuario abre /benchmark en el navegador

2. Frontend carga los modelos disponibles (GET /models)
   └─> Muestra checkboxes con los modelos instalados en Ollama

3. Usuario selecciona modelos y modo (Rápido / Completo)

4. Clic en "Iniciar benchmark"
   └─> Frontend llama POST /benchmark/start
   └─> Backend genera job_id y lanza thread

5. Thread de benchmark:
   a. Lee test_questions.json
   b. Si quick=true: selecciona 1 pregunta por categoría (máx. 6)
   c. Carga ChromaDB y modelo de embeddings (MiniLM)
   d. Para cada modelo:
      - Crea cadena RAG (LangChain LCEL)
      - Para cada pregunta:
        i.   Mide memoria antes
        ii.  Ejecuta query_rag() y mide latencia
        iii. Mide memoria después
        iv.  Calcula las 5 métricas
        v.   Actualiza job en memoria (visible via polling)
   e. Guarda resultados en benchmark/results/

6. Frontend hace polling cada 2 segundos (GET /benchmark/progress/{job_id})
   └─> Actualiza barra de progreso
   └─> Actualiza tarjetas de métricas en tiempo real
   └─> Muestra modelo que se está procesando actualmente

7. Al terminar (status: "done"):
   └─> Muestra tabla comparativa final
   └─> Permite expandir respuestas individuales por pregunta
```

---

## Interfaz del frontend (`/benchmark`)

### Panel de configuración
- **Checkboxes de modelos**: carga automáticamente los modelos de Ollama disponibles. Se pueden seleccionar 2 o más para comparar.
- **Modo Rápido / Completo**: rápido usa 6 preguntas (1 por categoría), completo usa las 25.

### Panel de progreso (durante la ejecución)
- Barra de progreso con porcentaje
- Nombre del modelo que se está procesando actualmente
- Badges de estado por modelo: gris (pendiente), azul pulsante (procesando), verde (terminado)

### Tarjetas de métricas (en tiempo real)
Una tarjeta por cada métrica con:
- Barras comparativas por modelo
- Badge "✓ mejor" en el modelo ganador
- Indicador de si mayor o menor es mejor

### Tabla comparativa
Tabla con todos los modelos como columnas y todas las métricas como filas. Las celdas del modelo ganador en cada métrica aparecen resaltadas en verde.

### Acordeón de respuestas
Al terminar, se puede expandir un panel con todas las respuestas individuales mostrando: pregunta, respuesta, categoría, dificultad, si recuperó el documento correcto, relevancia y fidelidad.

### Ejecuciones anteriores
Si no hay un benchmark activo, la página muestra automáticamente los resultados de las últimas 10 ejecuciones guardadas en disco.

---

## Resultados guardados en disco

Cada ejecución genera dos archivos en `deploy/Prototipo/benchmark/results/`:

**`{timestamp}_summary.json`** — Métricas agregadas por modelo:
```json
{
  "qwen2.5:3b": {
    "total_questions": 6,
    "successful": 6,
    "avg_latency_seconds": 8.421,
    "retrieval_accuracy": 0.833,
    "avg_answer_relevancy": 0.712,
    "avg_faithfulness": 0.885,
    "hallucination_rate": 0.0,
    "avg_memory_mb": 12.4
  }
}
```

**`{timestamp}_raw_results.json`** — Resultado individual por pregunta:
```json
[
  {
    "question_id": "q001",
    "model_name": "qwen2.5:3b",
    "question": "Cuales son los derechos de los egresados...",
    "answer": "Según el Reglamento de Egresados...",
    "category": "reglamento_egresados",
    "difficulty": "easy",
    "latency_seconds": 7.234,
    "memory_usage_mb": 8.1,
    "retrieval_hit": true,
    "answer_relevancy": 0.741,
    "faithfulness": 0.9,
    "hallucination_detected": false,
    "no_answer_correct": true
  }
]
```

---

## Modelos disponibles para comparar

| Modelo | Parámetros | RAM aprox. | Velocidad estimada |
|---|---|---|---|
| `qwen2.5:1.5b` | 1.5B | ~1.5 GB | Muy rápido (~3-5s) |
| `qwen2.5:3b` | 3B | ~3.5 GB | Rápido (~7-12s) |
| `llama3.2:1b` | 1B | ~1.2 GB | Muy rápido (~3-5s) |
| `llama3.2:3b` | 3B | ~3.5 GB | Rápido (~7-12s) |
| `phi3:mini` | 3.8B | ~4 GB | Medio (~10-15s) |
| `mistral:7b` | 7B | ~7.5 GB | Lento (~20-40s) |

Para descargar un modelo adicional (CMD):
```cmd
ollama pull qwen2.5:1.5b
```

---

## Limitaciones conocidas

1. **Memoria RAM**: la métrica `avg_memory_mb` mide solo el proceso Python, no la VRAM/RAM de Ollama. Para medir el uso real del modelo hay que monitorear Ollama por separado.

2. **Detección de alucinaciones**: el método actual es heurístico (busca números nuevos). No detecta alucinaciones de tipo textual (frases inventadas que no contienen números).

3. **Relevancia como proxy**: la similitud coseno entre pregunta y respuesta es un indicador aproximado de relevancia, no una evaluación humana. Una respuesta correcta pero técnica puede tener score bajo.

4. **Concurrencia**: si se lanzan dos benchmarks simultáneos, compiten por la GPU/CPU de Ollama y los tiempos serán mayores de lo normal.
