#!/bin/bash
ls -la /dev/video* 2>&1 || true
ros2 pkg list | grep -E "orbbec|usb_cam|v4l2|image" || true
ping -c 1 -W 1 192.168.1.153
