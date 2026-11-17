"""Nodo ROS 2 de cámara Orbbec Astra Pro Plus por UVC (/dev/video0 o stream compartido).

Se utiliza automáticamente en `bringup.launch.py` cuando `camera:=orbbec` y el paquete
externo `orbbec_camera` no está instalado en el workspace, publicando:
  - `/camera/color/image_raw` (`sensor_msgs/msg/Image`, `rgb8`)
  - `/camera/color/camera_info` (`sensor_msgs/msg/CameraInfo`): intrínsecos calibrados de
    `intrinsics_file` si existe; si no, nominales Astra Pro Plus (HFOV 66.1°, sin distorsión)
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class OrbbecUvcNode(Node):
    def __init__(self):
        super().__init__('orbbec_uvc')
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('fallback_file', os.path.expanduser('~/.ros/lite6_handeye/.live_cam.jpg'))
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('hfov_deg', 66.1)
        self.declare_parameter('intrinsics_file', os.path.expanduser('~/.ros/lite6_handeye/orbbec_intrinsics.yaml'))
        self.declare_parameter('optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')

        self.dev = self.get_parameter('device').value
        self.fallback_file = self.get_parameter('fallback_file').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.hfov = math.radians(float(self.get_parameter('hfov_deg').value))
        self.optical_frame = self.get_parameter('optical_frame').value

        self.intr = None
        path = os.path.expanduser(self.get_parameter('intrinsics_file').value)
        if path and os.path.exists(path):
            with open(path) as f:
                self.intr = yaml.safe_load(f)
            self.get_logger().info(f'Intrínsecos calibrados de {path} '
                                   f'(RMS {self.intr.get("rms_px", float("nan")):.3f} px)')
        else:
            self.get_logger().warn('Sin intrinsics_file: usando intrínsecos nominales (HFOV)')

        img_topic = self.get_parameter('image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value

        self.pub_img = self.create_publisher(Image, img_topic, 10)
        self.pub_info = self.create_publisher(CameraInfo, info_topic, 10)

        self.cap = None
        if os.path.exists(self.dev):
            cap = cv2.VideoCapture(self.dev)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap = cap
                self.get_logger().info(f'Orbbec UVC abierto directamente en {self.dev} ({self.width}x{self.height})')

        if self.cap is None:
            self.get_logger().info(f'Usando puente UVC en vivo desde {self.fallback_file}')

        # puente por archivo: sondear ~4x más rápido que la cámara y publicar solo cuadros nuevos y
        # completos (ffmpeg reescribe el archivo en el lugar; leerlo a la misma frecuencia pierde cuadros)
        self._last_stamp = None
        period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(period if self.cap is not None else period / 4.0, self._tick)

    def _make_camera_info(self, w: int, h: int, stamp) -> CameraInfo:
        if self.intr is not None:
            K = self.intr['camera_matrix']
            s = w / float(self.intr['image_width'])    # escala si la resolución difiere
            fx, fy, cx, cy = K[0] * s, K[4] * s, K[2] * s, K[5] * s
            d = [float(v) for v in self.intr['distortion']]
        else:
            fx = w / (2.0 * math.tan(self.hfov / 2.0))
            fy = fx
            cx = w / 2.0
            cy = h / 2.0
            d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.optical_frame
        msg.width = w
        msg.height = h
        msg.distortion_model = 'plumb_bob'
        msg.d = d
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def _tick(self):
        bgr = None
        mtime_ns = None
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                bgr = frame
        elif os.path.exists(self.fallback_file):
            try:
                st = os.stat(self.fallback_file)
                stamp = (st.st_mtime_ns, st.st_size)
                if stamp == self._last_stamp:
                    return
                with open(self.fallback_file, 'rb') as f:
                    buf = f.read()
                if len(buf) > 256 and buf[:2] == b'\xff\xd8' and buf[-2:] == b'\xff\xd9':
                    arr = np.frombuffer(buf, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        self._last_stamp = stamp   # incompleto: se reintenta en el próximo sondeo
                        mtime_ns = st.st_mtime_ns  # hora en que ffmpeg escribió el cuadro (más cerca de la captura)
            except Exception:
                bgr = None

        if bgr is None:
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        stamp = self.get_clock().now().to_msg()
        if mtime_ns is not None:
            stamp.sec, stamp.nanosec = divmod(int(mtime_ns), 1_000_000_000)

        img_msg = Image()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.optical_frame
        img_msg.height = h
        img_msg.width = w
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = 0
        img_msg.step = w * 3
        img_msg.data = rgb.tobytes()

        self.pub_img.publish(img_msg)
        self.pub_info.publish(self._make_camera_info(w, h, stamp))


def main(args=None):
    rclpy.init(args=args)
    node = OrbbecUvcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap is not None:
            node.cap.release()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
