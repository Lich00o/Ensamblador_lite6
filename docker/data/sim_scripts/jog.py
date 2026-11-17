"""Jog manual del Lite 6 (articular y cartesiano) para el panel Tk.

Mientras un botón está presionado, un lazo a 50 Hz integra la consigna (con rampa de 0.3 s
al arrancar y al soltar) y la publica en /lite6_traj_controller/joint_trajectory con el mismo
formato que el tracker (punto +0.1 s con velocidad + punto de frenado a v = 0). Al soltar,
la rampa baja a 0 y se deja de publicar.

Cartesiano: X/Y/Z en link_base (m) y Rx/Ry/Rz alrededor de los ejes de link_base pasando por
el flange (link_eef); IK numérica sembrada en la consigna actual. Se detiene si la IK no
converge o pide un salto grande (singularidad / límite).
"""
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from builtin_interfaces.msg import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lite6_handeye.person_tracker import lite6_fk, lite6_ik_6dof

JOINTS = [f'joint{i}' for i in range(1, 7)]
RATE = 50.0
DT = 1.0 / RATE
RAMP_S = 0.3                   # tiempo de rampa de velocidad (arranque y frenado)
LOOKAHEAD = 0.10
V_JOINT = 0.6                  # rad/s al 100 %
V_LIN = 0.10                   # m/s al 100 %
V_ROT = 0.6                    # rad/s al 100 %
MARGIN = np.radians(1.0)       # margen a los límites del URDF


class Jogger:
    def __init__(self, node, on_start=None):
        self.node = node
        self.on_start = on_start          # p. ej. pausar el tracker al empezar a mover
        self.q_meas = None
        self.lower = np.radians([-178, -150, -3.5, -178, -124, -178])
        self.upper = np.radians([178, 150, 178, 178, 124, 178])
        self.speed = 0.25
        self.axis = None                  # ('j', 0..5) | ('x'|'y'|'z'|'rx'|'ry'|'rz', None)
        self.sign = 0
        self.heartbeat = 0.0
        self.s = 0.0                      # factor de rampa 0..1
        self.q_cmd = None
        self.T_cmd = None                 # pose cartesiana deseada (sin deriva de la IK)
        self.msg = ''
        self.robot_ok = lambda: True
        self.lock = threading.Lock()
        node.create_subscription(JointState, '/joint_states', self._on_js, 20)
        node.create_subscription(String, '/robot_description', self._on_urdf,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub = node.create_publisher(JointTrajectory, '/lite6_traj_controller/joint_trajectory', 10)
        threading.Thread(target=self._loop, daemon=True).start()

    def _on_js(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in JOINTS):
            self.q_meas = np.array([d[j] for j in JOINTS])

    def _on_urdf(self, m):
        root = ET.fromstring(m.data)
        for j in root.findall('joint'):
            if j.get('name') in JOINTS and j.find('limit') is not None:
                i = JOINTS.index(j.get('name'))
                self.lower[i] = float(j.find('limit').get('lower'))
                self.upper[i] = float(j.find('limit').get('upper'))

    # --- UI ---
    def press(self, axis, sign):
        if not self.robot_ok():
            self.msg = 'robot no listo: usa "Robot a mode 1"'
            return
        if self.q_meas is None:
            self.msg = 'sin /joint_states'
            return
        with self.lock:
            if self.axis is None and self.on_start:
                self.on_start()
            self.axis, self.sign, self.heartbeat = axis, sign, time.monotonic()
            self.msg = ''

    def release(self):
        with self.lock:
            self.sign = 0

    def keepalive(self):
        if self.sign:
            self.heartbeat = time.monotonic()

    def tcp(self):
        if self.q_meas is None:
            return None
        T = lite6_fk(self.q_meas)
        return T[:3, 3] * 1000.0, Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)

    # --- lazo 50 Hz ---
    def _publish(self, q, v):
        m = JointTrajectory()
        m.joint_names = JOINTS
        a = JointTrajectoryPoint(positions=[float(x) for x in q + v * LOOKAHEAD],
                                 velocities=[float(x) for x in v],
                                 time_from_start=Duration(nanosec=int(LOOKAHEAD * 1e9)))
        b = JointTrajectoryPoint(positions=[float(x) for x in q + v * (LOOKAHEAD + 0.1)],
                                 velocities=[0.0] * 6,
                                 time_from_start=Duration(nanosec=int((LOOKAHEAD + 0.2) * 1e9)))
        m.points = [a, b]
        self.pub.publish(m)

    def _step(self, axis, sign):
        q = self.q_cmd.copy()
        kind, j = axis
        if kind == 'j':
            q[j] += sign * V_JOINT * self.speed * self.s * DT
            if not (self.lower[j] + MARGIN <= q[j] <= self.upper[j] - MARGIN):
                self.msg = f'J{j + 1} en el límite'
                return None
            return q
        if self.T_cmd is None:
            self.T_cmd = lite6_fk(q)
        T = self.T_cmd.copy()
        if kind in ('x', 'y', 'z'):
            T[{'x': 0, 'y': 1, 'z': 2}[kind], 3] += sign * V_LIN * self.speed * self.s * DT
        else:
            ax = np.eye(3)[{'rx': 0, 'ry': 1, 'rz': 2}[kind]]
            R = Rotation.from_rotvec(ax * sign * V_ROT * self.speed * self.s * DT).as_matrix()
            T[:3, :3] = R @ T[:3, :3]
        q_new, ok = lite6_ik_6dof(T, q)
        if not ok or np.max(np.abs(q_new - q)) > 0.05 or np.any(q_new < self.lower + MARGIN) \
                or np.any(q_new > self.upper - MARGIN):
            self.msg = 'límite de alcance / singularidad: prueba otro eje'
            return None
        self.T_cmd = T
        return q_new

    def _loop(self):
        while True:
            t0 = time.monotonic()
            with self.lock:
                axis, sign = self.axis, self.sign
                if sign and time.monotonic() - self.heartbeat > 0.4:
                    sign = self.sign = 0              # la UI dejó de confirmar: soltar
            if axis is not None:
                if self.q_cmd is None:
                    self.q_cmd, self.s, self.T_cmd = self.q_meas.copy(), 0.0, None
                self.s = min(1.0, self.s + DT / RAMP_S) if sign else max(0.0, self.s - DT / RAMP_S)
                if not self.robot_ok():
                    self.s = 0.0
                q = self._step(axis, sign or self._last_sign) if self.s > 0 else None
                if q is None:
                    self._publish(self.q_cmd, np.zeros(6))
                    with self.lock:
                        if not self.sign or self.s == 0.0 or self.msg:
                            self.axis, self.q_cmd, self.sign, self.T_cmd = None, None, 0, None
                else:
                    v = (q - self.q_cmd) / DT
                    self.q_cmd = q
                    self._publish(q, v)
                if sign:
                    self._last_sign = sign
            time.sleep(max(0.0, DT - (time.monotonic() - t0)))

    _last_sign = 1
