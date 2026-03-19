# Guía de Despliegue — Cluster Físico Universitario

Guía paso a paso para desplegar el Asistente RAG + SLM Uninorte en un nodo del cluster físico del laboratorio.

---

## Índice

1. [Checklist de requisitos del nodo](#1-checklist-de-requisitos-del-nodo)
2. [Acceso SSH al servidor](#2-acceso-ssh-al-servidor)
3. [Verificación del hardware](#3-verificación-del-hardware)
4. [Preparación del sistema](#4-preparación-del-sistema)
5. [Instalación de Ollama y el modelo](#5-instalación-de-ollama-y-el-modelo)
6. [Clonar el repositorio](#6-clonar-el-repositorio)
7. [Configurar el backend (Python + FastAPI)](#7-configurar-el-backend-python--fastapi)
8. [Construir el frontend (Next.js)](#8-construir-el-frontend-nextjs)
9. [Configurar servicios con systemd](#9-configurar-servicios-con-systemd)
10. [Configurar Nginx como reverse proxy](#10-configurar-nginx-como-reverse-proxy)
11. [Abrir el firewall](#11-abrir-el-firewall)
12. [Verificación final](#12-verificación-final)
13. [Comandos de mantenimiento](#13-comandos-de-mantenimiento)
14. [Solución de problemas](#14-solución-de-problemas)

---

## 1. Checklist de requisitos del nodo

Antes de empezar, confirmar con el profesor que el nodo cumple lo siguiente:

| Requisito | Mínimo | Recomendado | Cómo verificar |
|-----------|--------|-------------|----------------|
| CPU cores | 4 cores | 8+ cores | `nproc` |
| Arquitectura | x86_64 | x86_64 | `uname -m` |
| Soporte AVX2 | Obligatorio | — | `grep -c avx2 /proc/cpuinfo` |
| RAM | 8 GB | **16 GB** | `free -h` |
| Disco libre | 20 GB | 50 GB SSD | `df -h /` |
| Sistema operativo | Ubuntu 20.04+ | **Ubuntu 22.04 LTS** | `lsb_release -a` |
| Acceso SSH | Obligatorio | — | — |
| IP en red universitaria | Obligatorio | IP fija preferida | `hostname -I` |
| Acceso sudo | Obligatorio | — | `sudo whoami` |

> **Pregunta exacta para el profesor:**
> *"¿El nodo tiene al menos 16 GB RAM, 4 cores x86_64 con AVX2, Ubuntu 22.04, acceso SSH, y una IP accesible desde la red de la universidad?"*

---

## 2. Acceso SSH al servidor

El profesor debe proporcionar:
- **IP del nodo** (ej: `192.168.1.50`)
- **Usuario** (ej: `ubuntu`, `student`, o el nombre asignado)
- **Contraseña o llave SSH** (`.pem` o `.pub`)

### Conectarse desde Windows (PowerShell o WSL)

```bash
# Con contraseña
ssh usuario@192.168.1.50

# Con llave SSH (.pem)
ssh -i C:\ruta\a\llave.pem usuario@192.168.1.50

# Si pide agregar al known_hosts, escribe "yes"
```

### Conectarse desde Linux/Mac

```bash
ssh -i /ruta/a/llave.pem usuario@192.168.1.50
```

---

## 3. Verificación del hardware

Una vez conectado por SSH, ejecutar estos comandos para confirmar que el nodo es suficiente:

```bash
# Arquitectura y SO
uname -m && lsb_release -d

# Verificar soporte AVX2 (debe devolver un número > 0)
grep -c avx2 /proc/cpuinfo

# RAM disponible (necesita al menos ~7 GB libres)
free -h

# Espacio en disco (necesita al menos 20 GB libres)
df -h /

# Cores de CPU
nproc

# Ver si hay GPU NVIDIA (opcional, acelera x5)
nvidia-smi 2>/dev/null || echo "Sin GPU NVIDIA detectada"
```

**Resultado esperado:**
```
x86_64
Description: Ubuntu 22.04.x LTS
4           ← avx2 (cualquier número > 0 es válido)
Mem:  15Gi   ← RAM total (mínimo 7Gi disponibles)
/dev/sda1  47G   15G  30G  34% /   ← suficiente espacio
8           ← cores
```

Si `avx2` devuelve `0`, el CPU **no soporta AVX2** y sentence-transformers fallará. Informar al profesor.

---

## 4. Preparación del sistema

Ejecutar como usuario con sudo:

```bash
# Actualizar paquetes del sistema
sudo apt-get update -y && sudo apt-get upgrade -y

# Instalar dependencias base
sudo apt-get install -y \
    curl git wget build-essential \
    python3.11 python3.11-venv python3-pip \
    nginx \
    netfilter-persistent iptables-persistent

# Instalar Node.js 20 LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verificar versiones
python3.11 --version   # Python 3.11.x
node --version          # v20.x.x
npm --version           # 10.x.x
nginx -v                # nginx/1.18.x
```

---

## 5. Instalación de Ollama y el modelo

```bash
# Instalar Ollama (detecta automáticamente x86_64 y GPU si hay)
curl -fsSL https://ollama.ai/install.sh | sh

# Iniciar el servicio
sudo systemctl enable ollama
sudo systemctl start ollama

# Esperar 3 segundos y verificar
sleep 3
ollama list

# Descargar el modelo principal (~2 GB, puede tardar 5-15 min)
ollama pull qwen2.5:3b

# Verificar que está instalado
ollama list
# Debe mostrar: qwen2.5:3b   ...   2.0 GB
```

> **Si hay GPU NVIDIA:** Ollama la detecta automáticamente. El modelo se carga en VRAM y la velocidad sube de ~10 tok/s a ~50 tok/s.

---

## 6. Clonar el repositorio

```bash
# Crear directorio de la aplicación
sudo mkdir -p /opt/uninorma
sudo chown $USER:$USER /opt/uninorma

# Clonar el branch Prototipo (donde está todo el código)
git clone \
    --branch Prototipo \
    https://github.com/Juanj01217/ProyectoFinal-SLM-UNINORMA.git \
    /opt/uninorma

# Verificar la estructura
ls /opt/uninorma
# Debe mostrar: Disenho/  deploy/  README.md  .git/
```

---

## 7. Configurar el backend (Python + FastAPI)

```bash
cd /opt/uninorma/Disenho/Prototipo

# Crear entorno virtual Python
python3.11 -m venv venv

# Activar el entorno e instalar dependencias
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verificar instalación crítica
python -c "import chromadb; import sentence_transformers; import fastapi; print('OK')"
# Debe imprimir: OK

# Ejecutar la ingesta de documentos
# (descarga PDFs de uninorte.edu.co y construye la base vectorial)
# Tarda ~3-10 minutos dependiendo de la conexión
python ingest.py --download

# Verificar resultado
ls data/chroma_db/   # Debe contener archivos de ChromaDB
ls data/raw/         # Debe contener los PDFs descargados

# Probar el backend directamente (Ctrl+C para detener)
python api.py &
sleep 5
curl http://localhost:8000/health
kill %1
# Debe mostrar JSON con ollama_running: true
```

---

## 8. Construir el frontend (Next.js)

```bash
cd /opt/uninorma/Disenho/frontend

# Instalar dependencias de Node.js
npm ci

# Construir para producción
npm run build

# Verificar que el build fue exitoso
ls .next/
# Debe contener: server/  static/  ...
```

---

## 9. Configurar servicios con systemd

Los servicios systemd permiten que la app arranque automáticamente y se reinicie si falla.

```bash
# --- Servicio Backend (FastAPI) ---
sudo tee /etc/systemd/system/uninorma-backend.service > /dev/null << 'EOF'
[Unit]
Description=Uninorma RAG Backend (FastAPI)
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=REEMPLAZAR_CON_TU_USUARIO
WorkingDirectory=/opt/uninorma/Disenho/Prototipo
ExecStart=/opt/uninorma/Disenho/Prototipo/venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# --- Servicio Frontend (Next.js) ---
sudo tee /etc/systemd/system/uninorma-frontend.service > /dev/null << 'EOF'
[Unit]
Description=Uninorma Frontend (Next.js)
After=network.target uninorma-backend.service

[Service]
Type=simple
User=REEMPLAZAR_CON_TU_USUARIO
WorkingDirectory=/opt/uninorma/Disenho/frontend
ExecStart=/usr/bin/node node_modules/.bin/next start --port 3000
Restart=on-failure
RestartSec=5
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
EOF

# Reemplazar REEMPLAZAR_CON_TU_USUARIO con el usuario actual
CURRENT_USER=$(whoami)
sudo sed -i "s/REEMPLAZAR_CON_TU_USUARIO/$CURRENT_USER/g" \
    /etc/systemd/system/uninorma-backend.service \
    /etc/systemd/system/uninorma-frontend.service

# Activar y arrancar los servicios
sudo systemctl daemon-reload
sudo systemctl enable uninorma-backend uninorma-frontend
sudo systemctl start uninorma-backend
sleep 5
sudo systemctl start uninorma-frontend

# Verificar estado
sudo systemctl status uninorma-backend --no-pager
sudo systemctl status uninorma-frontend --no-pager
```

Ambos deben mostrar **`Active: active (running)`**.

---

## 10. Configurar Nginx como reverse proxy

Nginx recibe las peticiones en el puerto 80 y las redirige al frontend (puerto 3000), que a su vez llama al backend (puerto 8000). El timeout de 180s es esencial para que el LLM pueda responder sin que Nginx corte la conexión.

```bash
# Copiar la configuración de Nginx del repositorio
sudo cp /opt/uninorma/deploy/nginx.conf /etc/nginx/sites-available/uninorma

# Activar el sitio
sudo ln -sf /etc/nginx/sites-available/uninorma /etc/nginx/sites-enabled/uninorma

# Desactivar el sitio default
sudo rm -f /etc/nginx/sites-enabled/default

# Verificar sintaxis
sudo nginx -t
# Debe decir: syntax is ok / test is successful

# Reiniciar Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

---

## 11. Abrir el firewall

```bash
# Si el nodo usa UFW (Ubuntu por defecto)
sudo ufw allow 22    # SSH (no bloquear)
sudo ufw allow 80    # HTTP (acceso a la app)
sudo ufw --force enable
sudo ufw status

# Si el nodo usa iptables directamente
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```

> **Nota importante:** Si la red del cluster tiene un firewall propio (común en universidades), el profesor deberá abrir el puerto 80 también en ese firewall externo. Sin esto, la app solo será accesible desde el mismo nodo.

---

## 12. Verificación final

### Desde el mismo servidor

```bash
# Verificar que todos los servicios están corriendo
sudo systemctl is-active ollama uninorma-backend uninorma-frontend nginx
# Debe mostrar 4 líneas con "active"

# Probar el health endpoint
curl -s http://localhost/api/health | python3 -m json.tool
```

Respuesta esperada:
```json
{
    "ollama_running": true,
    "available_models": {"qwen2.5:3b": true},
    "active_model": "qwen2.5:3b",
    "vector_store_ready": true
}
```

### Desde otra PC en la misma red universitaria

1. Obtener la IP del nodo: `hostname -I | awk '{print $1}'`
2. Abrir en el navegador: `http://IP_DEL_NODO`
3. Debe cargar la interfaz del asistente
4. Hacer una consulta de prueba: *"¿Cuáles son los derechos de los egresados?"*
5. La respuesta debe llegar en 10-30 segundos con fuentes citadas

---

## 13. Comandos de mantenimiento

```bash
# Ver estado de todos los servicios
sudo systemctl status ollama uninorma-backend uninorma-frontend nginx

# Ver logs en tiempo real
sudo journalctl -u uninorma-backend -f    # Backend
sudo journalctl -u uninorma-frontend -f   # Frontend

# Reiniciar un servicio
sudo systemctl restart uninorma-backend
sudo systemctl restart uninorma-frontend

# Reiniciar todo
sudo systemctl restart ollama uninorma-backend uninorma-frontend nginx

# Actualizar la app cuando hay cambios en el repo
sudo bash /opt/uninorma/deploy/update.sh

# Monitorear uso de RAM (importante: el modelo ocupa ~3.5 GB)
watch -n 2 free -h

# Ver qué procesos usan más memoria
ps aux --sort=-%mem | head -15

# Ver qué modelos tiene Ollama cargados en memoria
ollama ps
```

---

## 14. Solución de problemas

### La app no carga en el navegador (`connection refused`)

```bash
# 1. Verificar que nginx está corriendo
sudo systemctl status nginx

# 2. Verificar que el puerto 80 está abierto
sudo ss -tlnp | grep :80

# 3. Verificar firewall
sudo ufw status
# o
sudo iptables -L INPUT -n | grep 80
```

---

### El backend no responde (`/api/health` falla)

```bash
# Ver el log de errores del backend
sudo journalctl -u uninorma-backend -n 50 --no-pager

# Verificar que el puerto 8000 está ocupado por el backend
sudo ss -tlnp | grep :8000

# Verificar que Ollama está corriendo
sudo systemctl status ollama
ollama list   # Debe mostrar qwen2.5:3b
```

---

### Error: `Illegal instruction (core dumped)` al iniciar el backend

El CPU no tiene soporte AVX2, requerido por PyTorch/sentence-transformers.

```bash
grep -c avx2 /proc/cpuinfo   # Si devuelve 0, el CPU es incompatible
```

**Solución:** Pedir al profesor otro nodo con CPU más moderno (Intel Haswell 2013+ o AMD Ryzen).

---

### Primera consulta muy lenta (>60 segundos) o timeout

Normal en la primera consulta: Ollama carga el modelo en RAM (~3.5 GB). Las consultas siguientes son más rápidas.

```bash
# Verificar que el modelo está respondiendo
curl -s http://localhost:11434/api/tags | python3 -m json.tool

# Pre-cargar el modelo en memoria (opcional)
ollama run qwen2.5:3b "hola" --nowordwrap
```

---

### Error de memoria (OOM — Out of Memory)

```bash
# Ver cuánta RAM está disponible
free -h

# Ver qué ocupa más memoria
ps aux --sort=-%mem | head -10
```

Si hay menos de 7 GB libres al arrancar, el sistema puede matar procesos.

**Soluciones:**
- Cerrar otros procesos pesados en el nodo
- Usar el modelo más pequeño: cambiar `DEFAULT_SLM_MODEL = "qwen2.5:1.5b"` en `config.py` (solo necesita ~1.5 GB)
- Si el nodo tiene 8 GB y la RAM es muy justa, agregar swap:

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

### Reinstalación completa desde cero

Si algo sale mal y quieres empezar de nuevo:

```bash
sudo systemctl stop uninorma-backend uninorma-frontend
sudo systemctl disable uninorma-backend uninorma-frontend
sudo rm /etc/systemd/system/uninorma-backend.service
sudo rm /etc/systemd/system/uninorma-frontend.service
sudo rm -rf /opt/uninorma
sudo systemctl daemon-reload
# Luego volver al Paso 6
```
