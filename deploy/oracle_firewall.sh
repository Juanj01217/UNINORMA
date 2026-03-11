#!/bin/bash
# oracle_firewall.sh — Abrir puertos en el firewall de Ubuntu (iptables)
# Oracle Cloud bloquea por defecto todos los puertos excepto 22.
# Ejecutar como root después de crear la VM.

# Abrir puerto 80 (HTTP — Nginx)
iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT

# Guardar reglas para que persistan tras reinicios
netfilter-persistent save

echo "Puerto 80 abierto. También debes abrir el puerto 80 en la Security List de Oracle Cloud."
echo "Ve a: Networking → Virtual Cloud Networks → tu VCN → Security Lists → Ingress Rules"
echo "Agrega: Source 0.0.0.0/0, TCP, Destination Port 80"
