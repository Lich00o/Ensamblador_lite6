#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash

cleanup() {
    pkill -f "charuco_detector" 2>/dev/null || true
    pkill -f "auto_calibrator" 2>/dev/null || true
    pkill -f "sim_gazebo.launch.py" 2>/dev/null || true
    pkill -f "auto_calibrate.launch.py" 2>/dev/null || true
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f "parameter_bridge" 2>/dev/null || true
    pkill -f "move_group" 2>/dev/null || true
    pkill -f "robot_state_publisher" 2>/dev/null || true
}
cleanup
sleep 2

echo "=== 1. Lanzando sim_gazebo.launch.py (headless) ==="
ros2 launch lite6_handeye sim_gazebo.launch.py gui:=false rviz:=false > /tmp/sim_gz.log 2>&1 &
SIM_PID=$!

echo "=== 2. Esperando a que el robot llegue a READY y detecte el tablero ==="
for i in $(seq 1 60); do
    if grep -q "Succeeded" /tmp/sim_gz.log 2>/dev/null; then
        break
    fi
    sleep 1
done
sleep 2

python3 /home/ubuntu/.ros/lite6_handeye/sim_scripts/snap.py

echo "=== 3. Ejecutando auto_calibrate (20 poses en Gazebo) ==="
rm -f /home/ubuntu/.ros/lite6_handeye/lite6_zed_calib_poses.yaml \
      /home/ubuntu/.ros/lite6_handeye/lite6_zed_calib_samples.yaml \
      /home/ubuntu/.ros/lite6_handeye/lite6_zed_calib.yaml
bash /home/ubuntu/.ros/lite6_handeye/sim_scripts/run_gz_calib.sh /tmp/gzcalib.log

echo "=== 4. Log del calibrador ==="
grep -E "pose |muestra |rango de muñeca|poses alcanzables|RESULTADO|muestras usadas|consistencia|error|warn" /tmp/gzcalib.log || tail -n 30 /tmp/gzcalib.log

echo "=== 5. Evaluación contra Ground Truth (compare_gt) ==="
ros2 run lite6_handeye compare_gt

cleanup
