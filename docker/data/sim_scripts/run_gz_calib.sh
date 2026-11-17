#!/bin/bash
source /ws/install/setup.bash
pkill -f "lib/lite6_handeye/auto_calibrator"; sleep 1
LOG=${1:-/tmp/gzcalib.log}
ros2 launch lite6_handeye auto_calibrate.launch.py camera:=sim > "$LOG" 2>&1 &
for i in $(seq 1 30); do grep -q "Listo" "$LOG" 2>/dev/null && break; sleep 1; done
ros2 service call /handeye_auto_calibrator/start std_srvs/srv/Trigger >/dev/null 2>&1
for i in $(seq 1 1800); do grep -qE "RESULTADO|sin resultado|calibración abortada|abortando" "$LOG" && break; sleep 1; done
sleep 3
pkill -f "lite6_handeye auto_calibrate"; pkill -f "lib/lite6_handeye/auto_calibrator"
echo "=== FIN"
