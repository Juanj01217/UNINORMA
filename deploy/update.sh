#!/bin/bash
# update.sh — Actualizar la app desde el repositorio (ejecutar en el servidor)
set -e

APP_DIR="/opt/uninorma"
APP_USER="uninorma"

echo "Deteniendo servicios..."
systemctl stop uninorma-frontend uninorma-backend

echo "Actualizando código..."
cd $APP_DIR
sudo -u $APP_USER git pull origin Prototipo

echo "Actualizando dependencias Python..."
cd $APP_DIR/Disenho/Prototipo
sudo -u $APP_USER venv/bin/pip install -r requirements.txt --quiet

echo "Reconstruyendo frontend..."
cd $APP_DIR/Disenho/frontend
sudo -u $APP_USER npm ci --quiet
sudo -u $APP_USER npm run build

echo "Reiniciando servicios..."
systemctl start uninorma-backend
sleep 3
systemctl start uninorma-frontend

echo "Actualización completada."
systemctl status uninorma-backend --no-pager
systemctl status uninorma-frontend --no-pager
