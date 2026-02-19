## Ficha de Propuesta de Proyecto Final

**Ingeniería de Sistemas**
**202610**

Esta plantilla guía la formulación de temas de **Proyecto Final** con enfoque de **Diseño de Solución Tecnológica**. Se espera que cada propuesta describa un problema real y conduzca a una **experiencia mayor de diseño**: definición de alcance y objetivos, levantamiento de requerimientos, evaluación de alternativas, diseño de arquitectura e implementación de un producto funcional, incorporando **estándares de ingeniería** y **restricciones realistas** (tiempo, recursos, calidad, seguridad y operación), en línea con el lineamiento ABET.

---

## Título del tema

# APLICACCIONES DE SMALL LANGUAJE MODELS (SLM)

---

## Docente proponente

**EDUARDO ZUREK, PH.D.**

---

## Co-asesor(es) (opcional)

---

## Área / línea

**INTELIGENCIA ARTIFICIAL**

---

## Nº de estudiantes sugerido

**MÁXIMO 3 POR GRUPO**

---

## Descripción corta

*(Un párrafo “vendible”: qué problema, qué se construirá o investigará)*

Los Small Language Models (SLM) son modelos de lenguaje diseñados para comprender y generar texto en lenguaje natural con un número reducido de parámetros en comparación con los grandes modelos de lenguaje (LLM). Su objetivo principal no es cubrir conocimiento general masivo, sino ofrecer eficiencia, especialización y facilidad de despliegue, manteniendo un desempeño adecuado en tareas bien definidas.

Gracias a su tamaño compacto, los SLM requieren menos recursos computacionales, presentan menor latencia y pueden ejecutarse localmente en servidores institucionales, dispositivos edge o sistemas embebidos, lo que los convierte en una alternativa estratégica en contextos donde el costo, la privacidad de los datos y el consumo energético son factores críticos.

La importancia de los SLM radica en que permiten democratizar el uso de modelos de lenguaje en escenarios reales de producción, evitando la dependencia de infraestructuras en la nube y reduciendo riesgos asociados a la transferencia de información sensible. Además, su entrenamiento suele estar enfocado en dominios específicos, lo que incrementa su precisión en tareas especializadas frente a modelos más grandes pero generalistas.

Un primer ejemplo de aplicación de los SLM se encuentra en la industria y el mantenimiento predictivo, donde pueden analizar reportes técnicos y registros de fallas para clasificar incidentes, generar resúmenes automáticos y asistir a los ingenieros con explicaciones en lenguaje natural sobre el estado de los equipos, todo ello de forma local y en tiempo real.

Un segundo ejemplo relevante es el uso de SLM en educación superior, donde pueden actuar como tutores virtuales de asignaturas específicas, generar preguntas de evaluación basadas en un temario cerrado o apoyar la retroalimentación automática de respuestas cortas, sin exponer datos académicos a servicios externos.

En conjunto, los Small Language Models representan una solución práctica y eficiente para integrar inteligencia artificial basada en lenguaje en sistemas técnicos, científicos y educativos con requerimientos bien delimitados.

---

## Objetivo

**Diseñar, Implementar, Entrenar y Desplegar un SML para una aplicación específica**

---

# Alcance propuesto

---

## 1) Alcance de Diseño

### Incluye

* Definir el caso de uso y la tarea NLP (p. ej., clasificación, extracción, QA cerrada, resumen).
* Levantar requisitos funcionales (inputs/outputs, idioma, longitud, formato) y no funcionales (latencia, memoria, privacidad, disponibilidad).
* Seleccionar enfoque: SLM base + fine-tuning vs distillation vs RAG ligero (si aplica).
* Diseñar el pipeline: datos → entrenamiento → evaluación → empaquetado → despliegue.
* Definir métricas de éxito (F1, exactitud, EM, BLEU/ROUGE, latencia, costo por inferencia).

### Entregables

* Documento de arquitectura (alto nivel + componentes).
* Especificación de dataset y criterios de calidad.
* Plan de evaluación y benchmarking.

---

## 2) Alcance de Implementación

### Incluye

* Construcción del pipeline de datos: ingesta, limpieza, anonimización, etiquetado (si aplica) y versionado.
* Implementación del modelo: tokenización, head de tarea, entrenamiento reproducible.
* Implementación de inferencias: batch y online (API).
* Registro de experimentos (MLflow/W&B o equivalente) y control de versiones.

### Entregables

* Repositorio con código (data + training + inference).
* Scripts reproducibles + documentación de ejecución.
* Prototipo funcional (CLI o servicio mínimo).

---

## 3) Alcance de Entrenamiento

### Incluye

* Selección de SLM candidato (p. ej., 0.3B–3B) y configuración de entrenamiento.
* Estrategias de eficiencia: LoRA/QLoRA, cuantización, early stopping.
* Partición de datos: train/val/test, control de fuga de información.
* Evaluación comparativa: baseline (reglas/ML clásico) vs SLM.
* Análisis de errores (error analysis) y mitigación (balanceo, data augmentation, prompt templates si aplica).

### Entregables

* Modelo entrenado (pesos + tokenizer) y “model card”.
* Informe de resultados: métricas, curva de aprendizaje, costos, limitaciones.
* Checklist de riesgos (sesgos, alucinación, seguridad, privacidad).

---

## 4) Alcance de Despliegue

### Incluye

* Empaquetado: Docker/conda, cuantización INT8/INT4 si aplica.
* Exposición vía API REST/gRPC o integración con app existente.
* Monitoreo básico: latencia, throughput, errores, drift (si hay logs).
* Pruebas: unitarias, carga (load test), regresión de modelo.
* Documentación para operación (runbook) y guía de uso.

### Entregables

* Servicio desplegado (on-prem o cloud) + CI/CD básico.
* Manual de usuario técnico + guía de endpoints.
* Reporte de pruebas de rendimiento y requisitos hardware.

---

# Límites explícitos (Out of scope) recomendados

Para proteger el proyecto, normalmente se deja por fuera:

* Entrenar “desde cero” un modelo grande (solo fine-tuning/distillation).
* Soporte multilingüe completo (si no es requisito).
* Moderación avanzada y seguridad tipo enterprise (a menos que sea prioridad).
* Entrenamiento continuo automático (MLOps completo) salvo que se contemple.

---

# Criterios de éxito sugeridos (ejemplos)

* **Calidad:** F1 ≥ 0.85 en test (o meta definida por el dominio).
* **Eficiencia:** latencia p95 ≤ 300 ms (CPU/GPU según contexto).
* **Recursos:** RAM ≤ X GB, modelo cuantizado ≤ X GB.
* **Privacidad:** datos sensibles anonimizados + despliegue local si aplica.

---

# Plantilla corta de “Alcance del proyecto”

* Definir caso de uso, requisitos y métricas.
* Preparar y versionar dataset (con control de calidad y anonimización).
* Seleccionar SLM base y estrategia de adaptación (LoRA/QLoRA).
* Entrenar y evaluar con benchmarking + análisis de errores.
* Empaquetar, cuantizar y desplegar como API/integración.
* Documentar y entregar modelo, código, reportes y guía operativa.

---

## Riesgos

Que no se puedan cumplir todos los objetivos por limitaciones en los recursos de hardware disponibles.

---
