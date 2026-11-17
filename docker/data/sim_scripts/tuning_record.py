#!/usr/bin/env python3
"""Grabadora para afinar el seguimiento de personas (fase A: robot quieto).

Guarda las imágenes de /camera/color/image_raw (JPEG, nombre = stamp en ns), /ufactory/joint_states
(150 Hz) y camera_info en ~/.ros/lite6_handeye/diag/<nombre>/. La fase actual la escribe
tuning_record_host.sh en diag/phase.txt; se dibuja sobre el video en http://localhost:8091.
Termina cuando phase.txt dice FIN (o con Ctrl+C).

Uso (en lite6_gpu): python3 tuning_record.py <nombre>
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState

DIAG = os.path.expanduser('~/.ros/lite6_handeye/diag')
PHASE = f'{DIAG}/phase.txt'
JOINTS = [f'joint{i}' for i in range(1, 7)]
latest = {'jpg': None}


class Mjpeg(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
        self.end_headers()
        try:
            while True:
                if latest['jpg'] is not None:
                    self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + latest['jpg'] + b'\r\n')
                time.sleep(0.07)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


def phase():
    try:
        return open(PHASE).read().strip()
    except OSError:
        return ''


class Rec(Node):
    def __init__(self, out):
        super().__init__('tuning_record')
        self.out = out
        os.makedirs(f'{out}/img', exist_ok=True)
        self.js_t, self.js_q, self.info, self.n_img = [], [], None, 0
        self.create_subscription(Image, '/camera/color/image_raw', self.on_img, qos_profile_sensor_data)
        self.create_subscription(JointState, '/ufactory/joint_states', self.on_js, 200)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.on_info, qos_profile_sensor_data)

    def on_js(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in JOINTS):
            self.js_t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            self.js_q.append([d[j] for j in JOINTS])

    def on_info(self, m):
        if self.info is None:
            self.info = {'K': list(m.k), 'D': list(m.d), 'width': m.width, 'height': m.height}

    def on_img(self, m):
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding == 'rgb8' else img
        ns = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        ph = phase()
        if ph and ph != 'FIN':
            cv2.imwrite(f'{self.out}/img/{ns}.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.n_img += 1
        dbg = bgr.copy()
        cv2.rectangle(dbg, (0, 0), (dbg.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(dbg, ph or 'esperando inicio...', (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(dbg, f'REC {self.n_img} cuadros', (dbg.shape[1] - 330, dbg.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        latest['jpg'] = cv2.imencode('.jpg', cv2.resize(dbg, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    def save(self):
        np.savez(f'{self.out}/joints.npz', t=np.array(self.js_t), q=np.array(self.js_q))
        json.dump({'camera_info': self.info, 'n_img': self.n_img,
                   'calibration_file': os.path.expanduser('~/.ros/lite6_handeye/lite6_zed_calib.yaml')},
                  open(f'{self.out}/meta.json', 'w'), indent=1)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else time.strftime('rec_%H%M%S')
    rclpy.init()
    n = Rec(f'{DIAG}/{name}')
    srv = ThreadingHTTPServer(('0.0.0.0', 8091), Mjpeg)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f'grabando en {n.out}; video http://localhost:8091', flush=True)
    started = False
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.02)
            ph = phase()
            started = started or (ph not in ('', 'FIN'))
            if started and ph == 'FIN':
                break
    except KeyboardInterrupt:
        pass
    n.save()
    print(f'listo: {n.n_img} imágenes, {len(n.js_t)} muestras de articulaciones', flush=True)


if __name__ == '__main__':
    main()
