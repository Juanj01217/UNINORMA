# Instrucciones de Despliegue — Asistente Normatividad Uninorte

## Requisitos del equipo

| Recurso | Mínimo | Recomendado |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 2 cores | 4 cores |
| Disco libre | 20 GB | 32 GB |
| SO | Linux / Windows con WSL2 / macOS | Linux |

> El sistema corre completamente offline. No requiere conexión a internet después de la primera ejecución.

---

## Requisito previo: Docker

Verificar que Docker está instalado:

```bash
docker --version
docker compose version
```

Si no está instalado:
- **Linux:** `curl -fsSL https://get.docker.com | sh`
- **Windows:** instalar WSL2 + Ubuntu, luego el comando de Linux dentro de Ubuntu
- **macOS:** instalar [Rancher Desktop](https://rancherdesktop.io) (gratuito, sin cuenta)

---

## Levantar el sistema

Desde esta carpeta (`Disenho/`), ejecutar:

```bash
docker compose up
```

**Primera ejecución:** descarga automáticamente el modelo qwen2.5:3b (~2 GB). Puede tardar 5-15 minutos dependiendo de la conexión. No interrumpir.

El sistema está listo cuando aparezca en los logs:

```
backend-1  | [3/3] ChromaDB encontrada y lista.
backend-1  | INFO: Application startup complete.
frontend-1 | ✓ Ready in ...ms
```

---

## Acceder a la aplicación

| Servicio | URL |
|---|---|
| Interfaz de chat | http://localhost:3000 |
| API (diagnóstico) | http://localhost:8000/health |
| Documentación API | http://localhost:8000/docs |

---

## Detener el sistema

```bash
# Detener sin borrar datos
Ctrl+C

# O desde otra terminal
docker compose down
```

---

## Solución de problemas

**El backend dice "Ollama no disponible" al arrancar**
Normal. El contenedor de Ollama tarda ~30 segundos en iniciar. El backend reintenta automáticamente.

**El frontend carga pero las consultas no responden**
Verificar que el backend está activo:
```bash
curl http://localhost:8000/health
# Debe retornar: {"ollama_running": true, ...}
```

**La primera consulta tarda mucho (~30-60 segundos)**
Normal. El modelo se carga en RAM en la primera consulta. Las siguientes son más rápidas (~8-15 segundos).

**Error "port already in use"**
Los puertos 3000, 8000 u 11434 están ocupados. Verificar con:
```bash
docker compose ps
```

---

## Recursos utilizados en ejecución

| Componente | RAM |
|---|---|
| Ollama + qwen2.5:3b | ~3.5 GB |
| Backend (FastAPI + ChromaDB) | ~600 MB |
| Frontend (Next.js) | ~200 MB |
| **Total** | **~4.5 GB** |
