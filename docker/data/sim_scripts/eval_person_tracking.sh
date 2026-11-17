#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash

cd /ws
colcon build --symlink-install --packages-select lite6_handeye
source /ws/install/setup.bash
colcon test --packages-select lite6_handeye --event-handlers console_direct+
colcon test-result --verbose

cleanup() {
    pkill -f "person_tracker" 2>/dev/null || true
    pkill -f "publish_calibration" 2>/dev/null || true
    pkill -f "charuco_detector" 2>/dev/null || true
    pkill -f "sim_gazebo.launch.py" 2>/dev/null || true
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f "parameter_bridge" 2>/dev/null || true
    pkill -f "move_group" 2>/dev/null || true
    pkill -f "robot_state_publisher" 2>/dev/null || true
}
cleanup
sleep 2

echo "=== 1. Lanzando Gazebo + Lite 6 + person_target ==="
ros2 launch lite6_handeye sim_gazebo.launch.py gui:=false rviz:=false > /tmp/sim_gz_track.log 2>&1 &

for i in $(seq 1 60); do
    if grep -q "Succeeded" /tmp/sim_gz_track.log 2>/dev/null; then
        break
    fi
    sleep 1
done
sleep 2

echo "=== 2. Publicando TF calibrada y lanzando person_tracker (6-DOF) ==="
ros2 launch lite6_handeye publish_calibration.launch.py > /tmp/pub_cal.log 2>&1 &
sleep 1
ros2 launch lite6_handeye person_tracker.launch.py camera:=sim > /tmp/person_tracker.log 2>&1 &

echo "=== 3. Evaluando seguimiento 6-DOF con persona en movimiento ==="
python3 /home/ubuntu/.ros/lite6_handeye/sim_scripts/test_person_tracking.py

echo "=== 4. Log de person_tracker ==="
cat /tmp/person_tracker.log

cleanup
