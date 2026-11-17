#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash

cleanup() {
    pkill -f "[w]alk_person.py" 2>/dev/null || true
    pkill -f "[p]erson_tracker" 2>/dev/null || true
    pkill -f "[p]ublish_calibration" 2>/dev/null || true
    pkill -f "[c]haruco_detector" 2>/dev/null || true
    pkill -f "[s]im_gazebo.launch.py" 2>/dev/null || true
    pkill -f "[r]viz2" 2>/dev/null || true
    pkill -f "[g]z sim" 2>/dev/null || true
    pkill -f "[p]arameter_bridge" 2>/dev/null || true
    pkill -f "[m]ove_group" 2>/dev/null || true
    pkill -f "[r]obot_state_publisher" 2>/dev/null || true
    pkill -f "[s]tatic_transform_publisher" 2>/dev/null || true
}
cleanup
sleep 1

USD_FILE="${1:-/home/ubuntu/.ros/lite6_handeye/person_usd/female_adult_business_02.usd}"
rm -rf /ws/build/lite6_handeye/sim/models/person_target/meshes \
       /ws/install/lite6_handeye/share/lite6_handeye/sim/models/person_target/meshes
cd /ws
colcon build --symlink-install --packages-select lite6_handeye
source /ws/install/setup.bash

ros2 launch lite6_handeye sim_gazebo.launch.py gui:=false rviz:=false person_usd:="$USD_FILE" > /tmp/test_usd_gz.log 2>&1 &
for i in $(seq 1 45); do
    if grep -q "Succeeded" /tmp/test_usd_gz.log 2>/dev/null; then
        break
    fi
    sleep 1
done
sleep 5

ros2 launch lite6_handeye publish_calibration.launch.py > /tmp/pub_cal.log 2>&1 &
ros2 launch lite6_handeye person_tracker.launch.py camera:=sim > /tmp/pt_usd.log 2>&1 &
python3 /home/ubuntu/.ros/lite6_handeye/sim_scripts/walk_person.py 5

cleanup
