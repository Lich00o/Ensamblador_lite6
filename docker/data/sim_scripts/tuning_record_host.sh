#!/bin/bash
# Guion de la grabación para afinar el seguimiento (fase A, robot quieto). Corre en el HOST:
# lanza la grabadora en lite6_gpu, anuncia cada fase en voz alta (espeak-ng) y la marca en
# docker/data/diag/phases.json (reloj del host = reloj del contenedor).
# Uso: docker/data/sim_scripts/tuning_record_host.sh [nombre]
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"           # docker/data
NAME=${1:-rec_$(date +%H%M%S)}
DIAG="$DIR/diag"
mkdir -p "$DIAG"
: > "$DIAG/phase.txt"
say() { echo ">>> $1"; (espeak-ng -v es-419 -s 150 "$1" >/dev/null 2>&1 &); }

# la grabadora puede estar ya abierta (vista previa en :8091 antes de empezar); si no, se lanza
if docker exec lite6_gpu pgrep -f "tuning_record.py" >/dev/null; then
  NAME=$(docker exec lite6_gpu grep -o 'diag/[^;]*' /tmp/tuning_record.log | head -1 | cut -d/ -f2)
else
  docker exec -d lite6_gpu bash -lc "source /ws/install/setup.bash; export ROS_DOMAIN_ID=42; \
    python3 -u ~/.ros/lite6_handeye/sim_scripts/tuning_record.py $NAME > /tmp/tuning_record.log 2>&1"
  sleep 3
fi
echo "Grabación: $NAME"
echo "Video con la instrucción: http://localhost:8091"

PHASES=(
  "PREPARACION|10|Prepárate. Párate frente a la cámara, a unos dos metros, de cuerpo entero."
  "QUIETO|20|Quédate quieto, de pie, mirando a la cámara."
  "LATERAL LENTO|30|Camina lento de lado a lado, de izquierda a derecha, a la misma distancia."
  "PROFUNDIDAD|30|Acércate y aléjate lento, entre uno y tres metros."
  "LATERAL RAPIDO|15|Ahora muévete rápido de lado a lado."
  "QUIETO CERCA|10|Quédate quieto a un metro y medio."
)
echo "[" > "$DIAG/$NAME.phases.json.tmp"
first=1
for p in "${PHASES[@]}"; do
  IFS='|' read -r label dur text <<< "$p"
  say "$text"
  t0=$(date +%s.%N)
  echo "$label" > "$DIAG/phase.txt"
  for ((s = dur; s > 0; s--)); do
    echo "$label  $s s" > "$DIAG/phase.txt"
    [ "$s" -eq 5 ] && [ "$label" != PREPARACION ] && say "cinco segundos"
    sleep 1
  done
  t1=$(date +%s.%N)
  [ $first -eq 1 ] || echo "," >> "$DIAG/$NAME.phases.json.tmp"
  first=0
  echo "{\"fase\": \"$label\", \"t0\": $t0, \"t1\": $t1}" >> "$DIAG/$NAME.phases.json.tmp"
done
echo "]" >> "$DIAG/$NAME.phases.json.tmp"
mv "$DIAG/$NAME.phases.json.tmp" "$DIAG/$NAME/phases.json" 2>/dev/null || mv "$DIAG/$NAME.phases.json.tmp" "$DIAG/$NAME.phases.json"
echo "FIN" > "$DIAG/phase.txt"
say "Listo. Grabación terminada."
sleep 3
docker exec lite6_gpu tail -2 /tmp/tuning_record.log
