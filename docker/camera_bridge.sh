#!/bin/bash
# Puente de la Orbbec (UVC, en el host) hacia el contenedor: ffmpeg reescribe docker/data/.live_cam.jpg
# (montado como ~/.ros/lite6_handeye/) y orbbec_uvc lo publica en ROS. MJPEG 1280x720 a 30 fps.
# Si la cámara se desconecta del USB (pasa con el cable moviéndose con el brazo), ffmpeg termina:
# el bucle espera a que vuelva a aparecer y lo relanza. Uso: docker/camera_bridge.sh  (detener: Ctrl+C)
cd "$(dirname "$0")"
DEV=${CAM_DEV:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0}
[ -e "$DEV" ] || DEV=/dev/video0
trap 'exit 0' INT TERM
while true; do
  until [ -e "$DEV" ]; do sleep 0.5; done
  echo "$(date +%T) cámara: $DEV"
  ffmpeg -loglevel error -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 \
    -i "$DEV" -q:v 3 -update 1 -y data/.live_cam.jpg
  echo "$(date +%T) ffmpeg terminó (¿cámara desconectada?); reintentando..."
  sleep 1
done
