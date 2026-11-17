#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
rm -rf /ws/build/lite6_handeye/sim/models/person_target/meshes
cd /ws
colcon build --symlink-install --packages-select lite6_handeye
source /ws/install/setup.bash

cleanup() {
    pkill -9 -f "[w]alk_person.py" 2>/dev/null || true
    pkill -9 -f "[p]erson_tracker" 2>/dev/null || true
    pkill -9 -f "[p]ublish_calibration" 2>/dev/null || true
    pkill -9 -f "[c]haruco_detector" 2>/dev/null || true
    pkill -9 -f "[s]im_gazebo.launch.py" 2>/dev/null || true
    pkill -9 -f "[r]viz2" 2>/dev/null || true
    pkill -9 -f "[g]z sim" 2>/dev/null || true
    pkill -9 -f "[r]uby.*gz" 2>/dev/null || true
    pkill -9 -f "[p]arameter_bridge" 2>/dev/null || true
    pkill -9 -f "[m]ove_group" 2>/dev/null || true
    pkill -9 -f "[r]obot_state_publisher" 2>/dev/null || true
    pkill -9 -f "[s]tatic_transform_publisher" 2>/dev/null || true
}
cleanup
sleep 2

USD_PATH="${2:-/home/ubuntu/.ros/lite6_handeye/person_usd/male_adult_construction_01.usd}"
echo "=== 1. Abriendo Gazebo GUI + RViz (demo.rviz) con $USD_PATH ==="
ros2 launch lite6_handeye sim_gazebo.launch.py gui:=true rviz:=false person_usd:="$USD_PATH" > /tmp/sim_gz_gui.log 2>&1 &

for i in $(seq 1 60); do
    if grep -q "Configured and activated lite6_traj_controller" /tmp/sim_gz_gui.log 2>/dev/null; then
        break
    fi
    sleep 1
done
sleep 5

ros2 launch lite6_handeye publish_calibration.launch.py > /tmp/pub_cal.log 2>&1 &
ros2 launch lite6_handeye person_tracker.launch.py camera:=sim > /tmp/person_tracker.log 2>&1 &
rviz2 -d /home/ubuntu/.ros/lite6_handeye/demo.rviz --ros-args -p use_sim_time:=true > /tmp/rviz_demo.log 2>&1 &

echo "=== 2. Animando persona en Gazebo (walk_person.py) ==="
nohup python3 /home/ubuntu/.ros/lite6_handeye/sim_scripts/walk_person.py "${1:-600}" > /tmp/walk_person.log 2>&1 &
sleep 24
tail -n 15 /tmp/person_tracker.log 2>/dev/null || true
