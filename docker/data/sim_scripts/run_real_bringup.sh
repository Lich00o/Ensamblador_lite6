#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
rm -rf /ws/build/lite6_handeye/sim/models/person_target/meshes
cd /ws
colcon build --symlink-install --packages-select lite6_handeye
source /ws/install/setup.bash

pkill -9 -f "[w]alk_person.py" 2>/dev/null || true
pkill -9 -f "[p]erson_tracker" 2>/dev/null || true
pkill -9 -f "[p]ublish_calibration" 2>/dev/null || true
pkill -9 -f "[c]haruco_detector" 2>/dev/null || true
pkill -9 -f "[s]im_gazebo.launch.py" 2>/dev/null || true
pkill -9 -f "[b]ringup.launch.py" 2>/dev/null || true
pkill -9 -f "[o]rbbec_uvc" 2>/dev/null || true
pkill -9 -f "[r]viz2" 2>/dev/null || true
pkill -9 -f "[g]z sim" 2>/dev/null || true
pkill -9 -f "[r]uby.*gz" 2>/dev/null || true
pkill -9 -f "[p]arameter_bridge" 2>/dev/null || true
pkill -9 -f "[m]ove_group" 2>/dev/null || true
pkill -9 -f "[r]obot_state_publisher" 2>/dev/null || true
pkill -9 -f "[s]tatic_transform_publisher" 2>/dev/null || true
pkill -9 -f "[r]os2_control_node" 2>/dev/null || true
pkill -9 -f "[x]arm_driver_node" 2>/dev/null || true
pkill -9 -f "[j]oint_state_publisher" 2>/dev/null || true
sleep 2

echo "=== Lanzando bringup.launch.py con Lite 6 real (192.168.1.153) + Cámara Orbbec ==="
# DETECTOR=true activa charuco_detector (solo para calibrar: consume ~1 núcleo).
# setsid: sesión propia, así un Ctrl+C en la terminal del tracker no mata el bringup
nohup setsid ros2 launch lite6_handeye bringup.launch.py \
    robot_ip:=192.168.1.153 \
    camera:=orbbec \
    start_detector:=${DETECTOR:-false} \
    calibration_file:=/home/ubuntu/.ros/lite6_handeye/lite6_zed_calib.yaml \
    > /tmp/real_bringup.log 2>&1 &

for i in $(seq 1 30); do
    if grep -q "Ready to take commands for planning group lite6" /tmp/real_bringup.log 2>/dev/null; then
        break
    fi
    sleep 1
done
sleep 3
# controlador apto para la referencia en tramos del person_tracker (interpolate_from_desired_state)
bash "$(dirname "$0")/jtc_streaming.sh"

python3 - << 'EOF'
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
import cv2
import numpy as np

rclpy.init()
node = Node('probe_real_bringup')
res = {}
node.create_subscription(JointState, '/joint_states', lambda m: res.setdefault('js', m), 5)
node.create_subscription(Image, '/camera/color/image_raw', lambda m: res.setdefault('img', m), 2)
node.create_subscription(Image, '/charuco_detector/debug_image', lambda m: res.setdefault('dbg', m), 2)

t0 = time.time()
while time.time() - t0 < 5.0 and len(res) < 3:
    rclpy.spin_once(node, timeout_sec=0.1)

if 'js' in res:
    js = res['js']
    pairs = [f"{n}={p:.4f} rad ({np.degrees(p):.1f} deg)" for n, p in zip(js.name, js.position)]
    print("JOINT_STATES OK:", ", ".join(pairs))
else:
    print("JOINT_STATES: sin mensaje")

if 'img' in res:
    im = res['img']
    print(f"CAMERA RGB OK: {im.width}x{im.height} encoding={im.encoding} frame_id={im.header.frame_id}")
else:
    print("CAMERA RGB: sin mensaje")

if 'dbg' in res:
    dbg = res['dbg']
    arr = np.frombuffer(dbg.data, np.uint8).reshape(dbg.height, dbg.width, -1)
    cv2.imwrite('/home/ubuntu/.ros/lite6_handeye/real_charuco_dbg.png', arr)
    print(f"CHARUCO DETECTOR OK: {dbg.width}x{dbg.height} -> guardado real_charuco_dbg.png")

node.destroy_node()
rclpy.try_shutdown()
EOF
