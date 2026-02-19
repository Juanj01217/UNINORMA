# ProyectoFinal-SLM-UNINORMA

# Asistente Virtual Basado en Small Language Model (SLM) para la Consulta de Normatividad de Uninorte

---

## 1. Introducción

El acceso eficiente a la normatividad institucional es un reto recurrente en entornos universitarios. Actualmente, la Universidad del Norte publica su marco normativo exclusivamente en formato digital (principalmente PDFs) en su [portal oficial de normatividad](https://www.uninorte.edu.co/web/sobre-nosotros/normatividad), abarcando reglamentos estudiantiles, estatutos y lineamientos académicos y administrativos. Esta dispersión y extensión de información complejizan la consulta ágil y precisa, generando retrasos y sobrecarga en las áreas administrativas.

En este contexto, se propone el diseño y desarrollo de un asistente virtual, basado en un Small Language Model (SLM) local y una arquitectura de Retrieval-Augmented Generation (RAG), que permita consultar la normatividad institucional en lenguaje natural de forma segura, privada y eficiente.

---

## 2. Planteamiento del Problema

La comunidad universitaria enfrenta dificultades para encontrar respuestas rápidas y exactas sobre normatividad institucional debido a:
- La gran cantidad de documentos independientes, de considerable extensión, en formatos no estructurados (PDF).
- La falta de un motor de búsqueda semántica que relacione una consulta en lenguaje natural con el fragmento normativo relevante.
- El tiempo perdido por estudiantes y personal al buscar manualmente entre los documentos, lo que a su vez incrementa las consultas repetitivas a las secretarías académicas.

**Problema principal:**  
*La carencia de un sistema automatizado que permita resolver consultas sobre la normatividad universitaria, basada exclusivamente en los documentos oficiales disponibles, impacta negativamente la autogestión y genera sobrecarga administrativa*.

---

## 3. Restricciones y Supuestos de Diseño

### 3.1. Restricciones

- **Restricción de alcance:**  
  El asistente solo consultará documentos de la URL oficial de normatividad de Uninorte (`https://www.uninorte.edu.co/web/sobre-nosotros/normatividad`).  
- **Recursos computacionales:**  
  El sistema debe ejecutarse en servidores locales con hardware limitado (sin depender de APIs externas como OpenAI/Google).
- **Privacidad y seguridad:**  
  No se permitirá la integración con servicios en la nube externos para procesamiento del lenguaje o almacenamiento.
- **Fuentes de datos:**  
  Solamente se utilizarán los documentos PDF vigentes descargados de la URL definida.
- **Alucinaciones:**  
  El sistema debe limitar sus respuestas al contenido recuperado. Si la respuesta no está en la normatividad, debe indicarlo explícitamente.
- (COMPLETAR si existe otra restricción específica derivada de requerimientos del tutor o de recursos disponibles).

### 3.2. Supuestos

- Los documentos descargados son oficiales, actualizados y públicos.
- Las consultas estarán en español y el modelo SLM seleccionado será ajustado para ese idioma.
- Es posible la extracción programática completa de texto legible de todos los PDFs.
- (COMPLETAR otros supuestos relevantes detectados por el equipo).

---

## 4. Alcance

### 4.1. Incluye

- Extracción automatizada y actualización de los documentos PDF de la página oficial de normatividad.
- Procesamiento semántico mediante técnicas de RAG empleando un SLM open source (ej. Llama 3, Phi-3).
- Motor de búsqueda interno que relacione una consulta con la sección normativa pertinente, citando el documento fuente.
- Interfaz de usuario simple de tipo chat (web), accesible desde un navegador de la red interna.
- Reporte y documentación del diseño, la arquitectura y los resultados de pruebas.

### 4.2. No incluye

- Integración con sistemas de autenticación de Uninorte.
- Gestión o validación jurídica de respuestas.
- Procesamiento de trámites administrativos (solo consultas informativas).
- Despliegue a escala institucional ni soporte a consultas fuera del marco de la normatividad oficial.

---

## 5. Objetivos

### 5.1. Objetivo general

Diseñar y desarrollar un prototipo de asistente virtual, basado en un Small Language Model (SLM) y arquitectura RAG, para facilitar la consulta automática y precisa de la normatividad institucional de Uninorte a través de lenguaje natural.

### 5.2. Objetivos específicos

- Investigar y seleccionar las tecnologías apropiadas de extracción de texto y vectorización para documentos PDF normativos.
- Configurar y adaptar un SLM open source capaz de ejecutarse localmente, compatible con español.
- Implementar un motor de búsqueda semántica y pipeline de RAG para consultas normativas.
- Desarrollar una interfaz de usuario básica para la interacción y visualización de resultados.
- Evaluar la precisión y utilidad del sistema usando conjuntos de consultas de prueba y criterios de aceptación previamente definidos.

---

## 6. Criterios de aceptación iniciales

- El sistema debe responder correctamente al menos el X% (COMPLETAR: definir métrica inicial, ej. 80%) de un set de consultas predefinidas validadas por usuarios internos.
- Cada respuesta debe incluir la referencia (documento, artículo, sección) que respalda la información suministrada.
- El sistema debe operar completamente offline dentro de la infraestructura local y no enviar datos sensibles fuera de la universidad.
- Debe manejar correctamente la situación “no respuesta” cuando la normativa no contemple la pregunta.
- (COMPLETAR: definir otros criterios prácticos sugeridos por el tutor/equipo).

---

## 7. Estado del arte y soluciones relacionadas

Soluciones similares han sido implementadas en el ámbito educativo, aunque generalmente empleando modelos en la nube. Destacan:
- **Chatbots FAQ institucionales** con RAG sobre bases documentales cerradas [1].
- **Asistentes para consulta de normativa académica** en instituciones como la Universidad Nacional de Colombia y la Universidad de los Andes [2]–[3].

El uso de SLMs locales representa una ventaja en privacidad y eficiencia de costos frente a LLMs de uso general ([1], [4]).

---

## 8. Plan de trabajo (tentativo)

- **Semana 6–7:** Levantamiento y análisis detallado de requisitos; definición técnica de las herramientas y métodos de extracción.
- **Semana 8:** Pruebas de extracción y vectorización; descarga y procesamiento inicial de documentos.
- **Semana 9–10:** Desarrollo del motor de búsqueda y conexión al SLM.
- **Semana 11:** Implementación de la interfaz web de usuario.
- **Semana 12:** Integración, pruebas internas y validación inicial del pipeline.
- **Semana 13–14:** Ajustes, recolección de datos de pruebas, mejora de criterios de aceptación.
- **Semana 15–16:** Elaboración y revisión del informe final; preparación para sustentación.

*(COMPLETAR: responsable asignado por tarea, hito interno, retroalimentación esperada).*

---
