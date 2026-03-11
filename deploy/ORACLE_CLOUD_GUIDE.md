# Guía de Despliegue — Oracle Cloud Free Tier

## ¿Qué obtienes gratis?
- 4 vCPUs ARM Ampere A1
- 24 GB RAM
- 200 GB disco
- Precio: $0/mes para siempre

---

## Paso 1: Crear cuenta Oracle Cloud

1. Ve a [cloud.oracle.com](https://cloud.oracle.com) → **Start for free**
2. Regístrate con tu correo universitario
3. **Importante**: pide una tarjeta de crédito para verificación pero NO cobra nada

---

## Paso 2: Crear la VM

1. En el dashboard de Oracle Cloud → **Compute → Instances → Create Instance**

2. Configuración:
   - **Name**: `uninorma-server`
   - **Image**: Ubuntu 22.04 (Canonical)
   - **Shape**: `VM.Standard.A1.Flex` (ARM — es el gratuito)
     - OCPUs: **4**
     - Memory: **24 GB**
   - **Boot volume**: 100 GB (dentro del límite gratuito)

3. **SSH Keys**:
   - Si tienes una llave SSH existente, pégala
   - Si no, descarga la que Oracle genera (guárdala bien)

4. Clic en **Create**. La VM tarda ~2 minutos en arrancar.

---

## Paso 3: Abrir puerto 80 en Oracle Cloud

Oracle tiene **dos capas** de firewall. Debes abrir el puerto en ambas.

### 3a. Security List (firewall de Oracle)
1. Ve a **Networking → Virtual Cloud Networks → tu VCN**
2. Clic en **Security Lists → Default Security List**
3. Clic en **Add Ingress Rules**
4. Configura:
   - Source CIDR: `0.0.0.0/0`
   - Protocol: TCP
   - Destination Port: `80`
5. Clic en **Add Ingress Rules**

### 3b. iptables (firewall del SO Ubuntu)
Después de conectarte por SSH, ejecuta:
```bash
sudo bash /opt/uninorma/deploy/oracle_firewall.sh
```

---

## Paso 4: Conectarte por SSH

```bash
# Desde tu laptop (Linux/Mac/WSL)
ssh -i /ruta/a/tu/llave.pem ubuntu@TU_IP_PUBLICA

# Desde Windows (PowerShell)
ssh -i C:\ruta\a\llave.pem ubuntu@TU_IP_PUBLICA
```

La IP pública la encuentras en **Compute → Instances → tu instancia → Public IP**.

---

## Paso 5: Instalar todo con un solo script

```bash
# En el servidor SSH
curl -fsSL https://raw.githubusercontent.com/Juanj01217/ProyectoFinal-SLM-UNINORMA/Prototipo/deploy/server_setup.sh | sudo bash
```

O si ya clonaste el repo:
```bash
sudo bash /opt/uninorma/deploy/server_setup.sh
```

El script instala y configura todo automáticamente:
- Python 3.11 + dependencias del backend
- Node.js 20 + build del frontend
- Ollama + modelo qwen2.5:3b (~2 GB de descarga)
- Ingesta de documentos normativos de Uninorte
- Servicios systemd (auto-inicio)
- Nginx como reverse proxy en el puerto 80

**Tiempo estimado**: 20-40 minutos (principalmente la descarga del modelo).

---

## Paso 6: Verificar

Una vez que termine el script, abre en el navegador:

```
http://TU_IP_PUBLICA
```

Para verificar el estado del backend:
```
http://TU_IP_PUBLICA/api/health
```

---

## Comandos de administración

```bash
# Ver estado de los servicios
sudo systemctl status uninorma-backend
sudo systemctl status uninorma-frontend

# Ver logs en tiempo real
sudo journalctl -u uninorma-backend -f
sudo journalctl -u uninorma-frontend -f

# Reiniciar
sudo systemctl restart uninorma-backend
sudo systemctl restart uninorma-frontend

# Actualizar después de hacer push al repo
sudo bash /opt/uninorma/deploy/update.sh
```

---

## Solución de problemas

**La app no carga en el navegador**
- Verifica que el puerto 80 esté abierto en la Security List de Oracle
- Ejecuta `oracle_firewall.sh` para abrir el puerto en Ubuntu
- Ejecuta `sudo systemctl status nginx` para ver si nginx está corriendo

**El backend responde lento la primera vez**
- Normal. La primera consulta carga el modelo en RAM (~3-4 GB). Las siguientes son más rápidas.

**Ollama no responde**
```bash
sudo systemctl status ollama
sudo systemctl restart ollama
ollama list  # Verificar que qwen2.5:3b está instalado
```

**El modelo no está disponible**
```bash
ollama pull qwen2.5:3b
```
