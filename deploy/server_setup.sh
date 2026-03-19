#!/bin/bash
# =============================================================================
# server_setup.sh — Setup completo para Oracle Cloud Free Tier (Ubuntu 22.04 ARM)
# Asistente RAG + SLM Uninorte
#
# Uso:
#   chmod +x server_setup.sh
#   sudo bash server_setup.sh
# =============================================================================

set -e  # Detener si hay error

REPO_URL="https://github.com/Juanj01217/ProyectoFinal-SLM-UNINORMA.git"
REPO_BRANCH="Prototipo"
APP_DIR="/opt/uninorma"
APP_USER="uninorma"

echo "============================================="
echo " Asistente RAG + SLM Uninorte — Servidor"
echo " Oracle Cloud Free Tier (Ubuntu 22.04 ARM)"
echo "============================================="

# -----------------------------------------------------------------------------
# 1. Sistema base
# -----------------------------------------------------------------------------
echo "[1/9] Actualizando sistema..."
apt-get update -y && apt-get upgrade -y
apt-get install -y curl git wget unzip build-essential nginx python3.11 python3.11-venv python3-pip

# -----------------------------------------------------------------------------
# 2. Node.js 20 LTS
# -----------------------------------------------------------------------------
echo "[2/9] Instalando Node.js 20 LTS..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
node --version && npm --version

# -----------------------------------------------------------------------------
# 3. Ollama (soporte nativo ARM64)
# -----------------------------------------------------------------------------
echo "[3/9] Instalando Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh
systemctl enable ollama
systemctl start ollama
sleep 3
echo "Descargando modelo qwen2.5:3b (~2 GB)..."
ollama pull qwen2.5:3b

# -----------------------------------------------------------------------------
# 4. Usuario y directorio de la aplicación
# -----------------------------------------------------------------------------
echo "[4/9] Creando usuario y directorio de la app..."
useradd -r -s /bin/bash -m -d /home/$APP_USER $APP_USER 2>/dev/null || true
mkdir -p $APP_DIR
chown -R $APP_USER:$APP_USER $APP_DIR

# -----------------------------------------------------------------------------
# 5. Clonar repositorio
# -----------------------------------------------------------------------------
echo "[5/9] Clonando repositorio (branch: $REPO_BRANCH)..."
if [ -d "$APP_DIR/.git" ]; then
    echo "Repositorio ya existe, actualizando..."
    cd $APP_DIR && sudo -u $APP_USER git pull origin $REPO_BRANCH
else
    sudo -u $APP_USER git clone --branch $REPO_BRANCH $REPO_URL $APP_DIR
fi

# -----------------------------------------------------------------------------
# 6. Backend Python — venv + dependencias
# -----------------------------------------------------------------------------
echo "[6/9] Configurando backend Python..."
cd $APP_DIR/Disenho/Prototipo
sudo -u $APP_USER python3.11 -m venv venv
sudo -u $APP_USER venv/bin/pip install --upgrade pip
sudo -u $APP_USER venv/bin/pip install -r requirements.txt

echo "Ejecutando ingesta de documentos (descarga PDFs de Uninorte)..."
sudo -u $APP_USER venv/bin/python ingest.py --download
echo "Ingesta completada."

# -----------------------------------------------------------------------------
# 7. Frontend Next.js — build de producción
# -----------------------------------------------------------------------------
echo "[7/9] Construyendo frontend Next.js..."
cd $APP_DIR/Disenho/frontend
sudo -u $APP_USER npm ci
sudo -u $APP_USER npm run build

# -----------------------------------------------------------------------------
# 8. Servicios systemd
# -----------------------------------------------------------------------------
echo "[8/9] Configurando servicios systemd..."

# Backend (FastAPI)
cat > /etc/systemd/system/uninorma-backend.service << EOF
[Unit]
Description=Uninorma RAG Backend (FastAPI)
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR/Disenho/Prototipo
ExecStart=$APP_DIR/Disenho/Prototipo/venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Frontend (Next.js)
cat > /etc/systemd/system/uninorma-frontend.service << EOF
[Unit]
Description=Uninorma Frontend (Next.js)
After=network.target uninorma-backend.service
Wants=uninorma-backend.service

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR/Disenho/frontend
ExecStart=/usr/bin/node node_modules/.bin/next start --port 3000
Restart=on-failure
RestartSec=5
Environment=NODE_ENV=production
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable uninorma-backend uninorma-frontend
systemctl start uninorma-backend
sleep 5
systemctl start uninorma-frontend

# -----------------------------------------------------------------------------
# 9. Nginx como reverse proxy
# -----------------------------------------------------------------------------
echo "[9/9] Configurando Nginx..."
cp $APP_DIR/deploy/nginx.conf /etc/nginx/sites-available/uninorma
ln -sf /etc/nginx/sites-available/uninorma /etc/nginx/sites-enabled/uninorma
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx
systemctl enable nginx

# -----------------------------------------------------------------------------
# Obtener IP pública
# -----------------------------------------------------------------------------
PUBLIC_IP=$(curl -s ifconfig.me || curl -s icanhazip.com)

echo ""
echo "============================================="
echo " DESPLIEGUE COMPLETADO"
echo "============================================="
echo ""
echo " URL de acceso:  http://$PUBLIC_IP"
echo " Backend API:    http://$PUBLIC_IP/api/health"
echo ""
echo " Comandos útiles:"
echo "   systemctl status uninorma-backend"
echo "   systemctl status uninorma-frontend"
echo "   journalctl -u uninorma-backend -f"
echo "   journalctl -u uninorma-frontend -f"
echo ""
