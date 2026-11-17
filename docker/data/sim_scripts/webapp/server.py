"""App web del tracker de personas + control del Lite 6.

  python3 server.py [--host 127.0.0.1] [--port 8080]      ->  http://localhost:8080

- Video MJPEG: /stream.mjpg?src=debug (cajas, IDs, objetivo) | src=raw (cámara).
- WebSocket /ws: estado a 10 Hz (robot, controlador, articulaciones, pose, tracks, objetivo) y
  comandos (seguimiento, vigilancia, robot listo, STOP, objetivo por ID o clic, jog, velocidad).
- El jog usa jog.Jogger (mismo formato de trayectoria que el tracker, rampas de 0.3 s) y se
  detiene si el navegador deja de confirmar la tecla presionada (keepalive) o se desconecta.

Por defecto escucha solo en 127.0.0.1: la página mueve el robot y no tiene autenticación.
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import uvicorn
from controller_manager_msgs.srv import ListControllers, SwitchController
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from geometry_msgs.msg import Point
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import SetBool, Trigger
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import SetInt16

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # jog.py vive en sim_scripts/
from jog import Jogger  # noqa: E402

from lite6_handeye.image_conv import image_to_numpy, to_bgr  # noqa: E402

NS = '/person_tracker'
CTRL = 'lite6_traj_controller'


class Frame:
    """Último cuadro de un topic de imagen, codificado a JPEG una sola vez (compartido)."""

    def __init__(self):
        self.msg, self.jpeg, self.seq, self.size, self.t = None, None, 0, (1280, 720), 0.0
        self.lock = threading.Lock()
        self.fps = 0.0

    def put(self, msg):
        now = time.monotonic()
        with self.lock:
            if self.t:
                self.fps = 0.9 * self.fps + 0.1 / max(now - self.t, 1e-3)
            self.msg, self.t, self.seq = msg, now, self.seq + 1
            self.size = (msg.width, msg.height)

    def jpeg_for(self, seq_seen):
        with self.lock:
            msg, seq = self.msg, self.seq
        if msg is None or seq == seq_seen:
            return None, seq_seen
        if self.jpeg is None or self.jpeg[0] != seq:
            raw, enc = image_to_numpy(msg)
            ok, buf = cv2.imencode('.jpg', to_bgr(raw, enc), [cv2.IMWRITE_JPEG_QUALITY, 75])
            self.jpeg = (seq, buf.tobytes())
        return self.jpeg[1], seq


class Bridge(Node):
    def __init__(self):
        super().__init__('tracker_web')
        self.frames = {'debug': Frame(), 'raw': Frame()}
        self.create_subscription(Image, f'{NS}/debug_image', self.frames['debug'].put, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/color/image_raw', self.frames['raw'].put, qos_profile_sensor_data)
        self.tracking, self.robot, self.tracks, self.target = None, None, [], None
        self.gallery = []
        self.ctrl_state, self.msg, self.msg_t = '?', '', 0.0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, f'{NS}/tracking_enabled', lambda m: setattr(self, 'tracking', m.data), latched)
        self.create_subscription(String, f'{NS}/tracks', self._on_tracks, 5)
        self.create_subscription(RobotMsg, '/ufactory/robot_states',
                                 lambda m: setattr(self, 'robot', (m.mode, m.state, m.err, m.warn)), 10)
        self.pub_sel = self.create_publisher(Int32, f'{NS}/select_target', 10)
        self.pub_click = self.create_publisher(Point, f'{NS}/debug_image_mouse_left', 10)
        self.cli = {
            'track': self.create_client(SetBool, f'{NS}/set_tracking'),
            'watch': self.create_client(Trigger, f'{NS}/go_watch'),
            'mode': self.create_client(SetInt16, '/ufactory/set_mode'),
            'state': self.create_client(SetInt16, '/ufactory/set_state'),
            'switch': self.create_client(SwitchController, '/controller_manager/switch_controller'),
            'list': self.create_client(ListControllers, '/controller_manager/list_controllers'),
        }
        self.jog = Jogger(self, on_start=self.pause_tracking)
        self.jog.robot_ok = self.robot_ok
        self.create_timer(1.0, self._poll_controller)

    # --- estado ---
    def _on_tracks(self, m):
        d = json.loads(m.data)
        self.tracks, self.target, self.gallery = d['tracks'], d['target'], d.get('gallery', [])

    def robot_ok(self):
        return self.robot is not None and self.robot[0] == 1 and self.robot[2] == 0 and self.robot[1] in (0, 1, 2)

    def _poll_controller(self):
        if self.cli['list'].service_is_ready():
            f = self.cli['list'].call_async(ListControllers.Request())
            f.add_done_callback(lambda f: setattr(self, 'ctrl_state', next(
                (c.state for c in f.result().controller if c.name == CTRL), 'ausente')))
        else:
            self.ctrl_state = 'sin controller_manager'

    def _stale_msg(self, now):
        raw, dbg = (now - self.frames[k].t > 1.0 for k in ('raw', 'debug'))
        if raw:
            return 'Sin imagen de cámara: revisa docker/camera_bridge.sh (host) y el nodo orbbec_uvc'
        if dbg:
            return 'person_tracker no publica: relanza pt_real.sh (la vista Cámara sí funciona)'
        return ''

    def note(self, text):
        self.msg, self.msg_t = text, time.monotonic()

    def state(self):
        tcp = self.jog.tcp()
        q = self.jog.q_meas
        r = self.robot
        now = time.monotonic()
        return {
            'tracking': self.tracking,
            'robot': None if r is None else {'mode': r[0], 'state': r[1], 'err': r[2], 'warn': r[3],
                                             'ok': self.robot_ok()},
            'controller': self.ctrl_state,
            'joints': None if q is None else [round(float(v), 1) for v in np.degrees(q)],
            'tcp': None if tcp is None else {'xyz': [round(float(v)) for v in tcp[0]],
                                             'rpy': [round(float(v), 1) for v in tcp[1]]},
            'tracks': self.tracks, 'target': self.target, 'gallery': self.gallery,
            # fps 0 si el topic dejó de publicar (> 1 s sin cuadros): así se ve un tracker caído
            'fps': {k: round(f.fps, 1) if now - f.t < 1.0 else 0.0 for k, f in self.frames.items()},
            'jogging': self.jog.axis is not None, 'speed': round(self.jog.speed * 100),
            'msg': self.jog.msg or (self.msg if now - self.msg_t < 6 else '') or self._stale_msg(now),
        }

    # --- comandos ---
    def _call(self, key, req, label):
        cli = self.cli[key]
        if not cli.service_is_ready():
            self.note(f'{label}: servicio no disponible (¿tracker / bringup corriendo?)')
            return
        f = cli.call_async(req)
        f.add_done_callback(lambda f: self.note(f'{label}: {getattr(f.result(), "message", "") or "ok"}'))

    def pause_tracking(self):
        if self.tracking:
            self._call('track', SetBool.Request(data=False), 'seguimiento')

    def ready(self):
        self._call('mode', SetInt16.Request(data=1), 'mode 1')
        self._call('state', SetInt16.Request(data=0), 'state 0')

        def activate():                  # el driver puede dejar el controlador inactivo
            time.sleep(1.5)
            self._call('switch', SwitchController.Request(activate_controllers=[CTRL], strictness=1,
                                                          activate_asap=True), 'controlador')
        threading.Thread(target=activate, daemon=True).start()

    def on_command(self, c):
        cmd = c.get('cmd')
        if cmd == 'tracking':
            self._call('track', SetBool.Request(data=bool(c['on'])), 'seguimiento')
        elif cmd == 'watch':
            self._call('watch', Trigger.Request(), 'vigilancia')
        elif cmd == 'ready':
            self.ready()
        elif cmd == 'stop':
            self.jog.release()
            self.pause_tracking()
            self._call('state', SetInt16.Request(data=4), 'STOP')
        elif cmd == 'select':
            self.pub_sel.publish(Int32(data=int(c['id'])))
        elif cmd == 'click':
            w, h = self.frames['debug'].size
            self.pub_click.publish(Point(x=float(c['x']) * w, y=float(c['y']) * h, z=0.0))
        elif cmd == 'jog':
            axis = ('j', int(c['axis'][1]) - 1) if c['axis'].startswith('j') else (c['axis'], None)
            self.jog.press(axis, int(c['sign']))
        elif cmd == 'jog_keep':
            self.jog.keepalive()
        elif cmd == 'jog_release':
            self.jog.release()
        elif cmd == 'speed':
            self.jog.speed = min(1.0, max(0.05, float(c['value']) / 100.0))


rclpy.init()
bridge = Bridge()
_ex = MultiThreadedExecutor(num_threads=4)
_ex.add_node(bridge)
threading.Thread(target=_ex.spin, daemon=True).start()

app = FastAPI(title='Lite 6 tracker')


@app.get('/')
def index():
    return FileResponse(os.path.join(HERE, 'index.html'))


@app.get('/stream.mjpg')
async def stream(src: str = 'debug'):
    frame = bridge.frames.get(src, bridge.frames['debug'])

    async def gen():
        seen = -1
        while True:
            jpg, seen2 = await asyncio.to_thread(frame.jpeg_for, seen)
            if jpg is None:
                await asyncio.sleep(0.02)
                continue
            seen = seen2
            yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + \
                b'\r\n\r\n' + jpg + b'\r\n'
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.websocket('/ws')
async def ws(sock: WebSocket):
    await sock.accept()

    async def push():
        while True:
            await sock.send_text(json.dumps(bridge.state()))
            await asyncio.sleep(0.1)
    task = asyncio.create_task(push())
    try:
        while True:
            bridge.on_command(json.loads(await sock.receive_text()))
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        bridge.jog.release()             # navegador cerrado: no dejar un jog activo


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 = accesible desde la red (sin autenticación)')
    ap.add_argument('--port', type=int, default=8080)
    a = ap.parse_args()
    print(f'App web: http://{"localhost" if a.host in ("127.0.0.1", "0.0.0.0") else a.host}:{a.port}', flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level='warning')
