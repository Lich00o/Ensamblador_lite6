#!/bin/bash
# Puente de la Orbbec (UVC /dev/video0, en el host) hacia el contenedor: ffmpeg reescribe
# docker/data/.live_cam.jpg (montado como ~/.ros/lite6_handeye/) y orbbec_uvc lo publica en ROS.
# La cámara entrega MJPEG 1280x720 a 30 fps. Uso: docker/camera_bridge.sh   (detener: Ctrl+C)
cd "$(dirname "$0")"
exec ffmpeg -loglevel error -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 \
  -i /dev/video0 -q:v 3 -update 1 -y data/.live_cam.jpg
