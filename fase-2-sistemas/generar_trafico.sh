#!/usr/bin/env bash

echo "============================================================"
echo "⚡ CLIENTE GENERADOR DE TRÁFICO CONTINUO (INFERENCIA EN VIVO)"
echo "============================================================"
echo "Enviando peticiones concurrentes a http://127.0.0.1:3000/predict..."
echo "Presiona [Ctrl + C] para detener el tráfico."
echo "------------------------------------------------------------"

contador=1
while true; do
  RESP=$(curl -s -X POST http://127.0.0.1:3000/predict \
       -H "Content-Type: application/json" \
       -d '{"features": [10.0, 5.0, 2.0]}')
  
  if [ -z "$RESP" ]; then
    echo "⚠️ Esperando a que el servidor esté activo en http://127.0.0.1:3000..."
    sleep 1
  else
    echo "  [Petición #$contador] 📥 $RESP"
    contador=$((contador + 1))
    sleep 0.25
  fi
done
