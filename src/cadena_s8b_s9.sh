#!/usr/bin/env bash
# S8b (búsqueda ampliada de la ResNet) y S9 (congelamiento) en una sola corrida desatendida.
# Apaga la VM al terminar, también si algo falla. Si S8b falla, S9 no corre.
#
#   bash src/cadena_s8b_s9.sh --prueba    # ensayo corto, sin apagar (~3 min)
#   bash src/cadena_s8b_s9.sh             # corrida completa (~2.5-3 h) y apagado
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/venv/bin/python}"

if [ "${1:-}" = "--prueba" ]; then
    PRUEBA="--prueba"
else
    PRUEBA=""
    trap 'echo "Apagando la VM en 1 minuto…"; sudo shutdown -h +1' EXIT
fi

echo "=== S8b: búsqueda ampliada de la ResNet ($(date -u)) ==="
"$PY" src/buscar.py --modelo resnet --variante b --L 2 3 4 --n-configs 25 \
    --lr-min 1e-5 --alpha-max 0.6 --filtros 16 32 \
    --bolsas red_activa+casi_positivo+nube_fria --perdidas bce bce+csi0.1 $PRUEBA \
  || { echo "S8b falló; S9 no se ejecuta."; exit 1; }

echo "=== S9: congelamiento, 6 configuraciones × 3 semillas ($(date -u)) ==="
"$PY" src/congelar.py $PRUEBA || { echo "S9 falló."; exit 1; }

echo "=== Cadena terminada ($(date -u)) ==="
