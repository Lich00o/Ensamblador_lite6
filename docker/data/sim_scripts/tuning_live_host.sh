#!/bin/bash
# Fase B del afinado (robot en movimiento). Corre en el HOST: lanza tuning_live.py en lite6_gpu
# y dice en voz alta cada instrucción que escribe en docker/data/diag/phase.txt.
# Requiere el tracker siguiendo a la persona (pt_real.sh true ...). Uso: tuning_live_host.sh [nombre]
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PH="$DIR/diag/phase.txt"
mkdir -p "$DIR/diag"
: > "$PH"
(
  last=""
  while true; do
    cur=$(cat "$PH" 2>/dev/null)
    if [ -n "$cur" ] && [ "$cur" != "$last" ]; then
      last="$cur"
      [ "$cur" = FIN ] && { espeak-ng -v es-419 -s 150 "Listo. Prueba terminada." >/dev/null 2>&1; break; }
      espeak-ng -v es-419 -s 150 "$cur" >/dev/null 2>&1
    fi
    sleep 0.3
  done
) &
SPEAK=$!
trap 'kill $SPEAK 2>/dev/null' EXIT
docker exec -it lite6_gpu bash -lc "source /ws/install/setup.bash; export ROS_DOMAIN_ID=42; \
  python3 -u ~/.ros/lite6_handeye/sim_scripts/tuning_live.py $*"
sleep 2
