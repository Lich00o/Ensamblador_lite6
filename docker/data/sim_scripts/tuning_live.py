#!/usr/bin/env python3
"""Fase B del afinado: recorre combinaciones de parámetros de movimiento en el person_tracker
en marcha (ros2 param set, sin reiniciar) y mide la vibración real de cada una.

Requiere el tracker siguiendo a una persona (pt_real.sh true ...). Por cada combinación: aplica
los parámetros, espera SETTLE s y graba REC s de /ufactory/joint_states (150 Hz) y del objetivo.
La instrucción para la persona se escribe en diag/phase.txt (tuning_live_host.sh la dice en voz
alta). Al terminar restaura los valores originales.

Métricas (mismas que tuning_replay.py + muñeca):
  vib_mm / vib_deg  TCP y eje óptico menos su media móvil de 1 s (RMS)
  muneca_deg        oscilación rápida de J4+J6 (posición menos media móvil de 0.2 s, RMS)
  err_deg           ángulo eje óptico -> objetivo filtrado del nodo
  arranques/min, vel pico J1 (°/s)
Uso (en lite6_gpu): python3 tuning_live.py [nombre]
"""
import json
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lite6_handeye import calib_io
from lite6_handeye.person_tracker import lite6_fk

DIAG = os.path.expanduser('~/.ros/lite6_handeye/diag')
SETTLE, REC = 4.0, 25.0
CONFIGS = [
    ('base', {}),
    ('acc 0.5', {'max_joint_acc_rad_s2': 0.5}),
    ('acc 1.5', {'max_joint_acc_rad_s2': 1.5}),
    ('jerk 3', {'max_joint_jerk_rad_s3': 3.0}),
    ('jerk 15', {'max_joint_jerk_rad_s3': 15.0}),
    ('vel 0.8', {'max_joint_vel_rad_s': 0.8}),
    ('lookahead 0.15', {'stream_lookahead_s': 0.15}),
    ('latencia 0.08', {'image_latency_s': 0.08}),
    ('base (repetición)', {}),
]
SAY = 'Camina lento de lado a lado, siempre igual.'
JOINTS = [f'joint{i}' for i in range(1, 7)]
X = calib_io.tf_dict_to_T(calib_io.load_result(os.path.expanduser(
    '~/.ros/lite6_handeye/lite6_zed_calib.yaml'))['eef_to_optical'])


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1, 1))))


def movavg(x, n):
    k = np.ones(n)
    num = np.apply_along_axis(lambda c: np.convolve(c, k, 'same'), 0, x)
    return num / np.convolve(np.ones(len(x)), k, 'same')[:, None]


def pval(v):
    if isinstance(v, bool):
        return ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=v)
    return ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))


class Live(Node):
    def __init__(self):
        super().__init__('tuning_live')
        self.js, self.tgt, self.rec = [], [], False
        self.create_subscription(JointState, '/ufactory/joint_states', self.on_js, 300)
        self.create_subscription(PointStamped, '/person_tracker/target_point', self.on_tgt, 50)
        self.set_cli = self.create_client(SetParameters, '/person_tracker/set_parameters')
        self.get_cli = self.create_client(GetParameters, '/person_tracker/get_parameters')

    def on_js(self, m):
        if self.rec:
            d = dict(zip(m.name, m.position))
            self.js.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, [d[j] for j in JOINTS]))

    def on_tgt(self, m):
        if self.rec:
            self.tgt.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.point.x, m.point.y, m.point.z))

    def call(self, cli, req):
        cli.wait_for_service(timeout_sec=5.0)
        f = cli.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=5.0)
        return f.result()

    def get(self, names):
        r = self.call(self.get_cli, GetParameters.Request(names=names))
        return {n: (v.double_value if v.type == ParameterType.PARAMETER_DOUBLE else v.bool_value)
                for n, v in zip(names, r.values)}

    def set(self, prm):
        if prm:
            r = self.call(self.set_cli, SetParameters.Request(
                parameters=[Parameter(name=k, value=pval(v)) for k, v in prm.items()]))
            bad = [res.reason for res in r.results if not res.successful]
            if bad:
                raise RuntimeError(f'parámetros rechazados: {bad}')

    def spin_for(self, s):
        t0 = time.monotonic()
        while time.monotonic() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.01)


def metrics(js, tgt):
    t = np.array([x[0] for x in js])
    q = np.array([x[1] for x in js])
    tu = np.arange(t[0], t[-1], 1 / 150)
    qu = np.stack([np.interp(tu, t, q[:, k]) for k in range(6)], 1)
    T = np.array([lite6_fk(qq) for qq in qu[::3]])            # 50 Hz para la pose
    pos, C = T[:, :3, 3], T @ X
    axis, cpos = C[:, :3, 2], C[:, :3, 3]
    vib = np.linalg.norm(pos - movavg(pos, 51), axis=1) * 1000
    vax = np.array([ang(a, b) for a, b in zip(axis, movavg(axis, 51))])
    w = np.degrees(qu[:, [3, 5]]) - movavg(np.degrees(qu[:, [3, 5]]), 31)
    err = np.nan
    if len(tgt) > 10:
        tt = np.array([x[0] for x in tgt])
        P = np.array([x[1:] for x in tgt])
        t50 = tu[::3]
        m = (t50 >= tt[0]) & (t50 <= tt[-1])
        tr = np.stack([np.interp(t50[m], tt, P[:, k]) for k in range(3)], 1)
        err = float(np.mean([ang(a, g - c) for a, g, c in zip(axis[m], tr, cpos[m])]))
    q50 = qu[::3]
    mv = np.r_[False, np.max(np.abs(np.diff(q50, axis=0)), axis=1) > 1e-5]
    dur = tu[-1] - tu[0]
    v1 = np.degrees(np.abs(np.gradient(qu[:, 0], tu)))
    return dict(vib_mm=float(np.sqrt(np.mean(vib ** 2))), vib_deg=float(np.sqrt(np.mean(vax ** 2))),
                muneca_deg=float(np.sqrt(np.mean(w[20:-20] ** 2))), err_deg=err,
                arranques_min=float(np.sum(mv[1:] & ~mv[:-1]) / dur * 60), vel_j1=float(np.percentile(v1, 99.5)))


def phase(txt):
    with open(f'{DIAG}/phase.txt', 'w') as f:
        f.write(txt)


def main():
    global CONFIGS
    args = sys.argv[1:]
    if args and args[0] == '--ab':
        # A/B alternado: base vs candidata (k=v ...), 3 veces cada una, para separar el efecto del ruido
        cand = {k: float(v) for k, v in (x.split('=') for x in args[1:] if '=' in x)}
        args = [x for x in args[1:] if '=' not in x]
        CONFIGS = [c for _ in range(3) for c in (('A base', {}), ('B candidata', cand))]
    name = args[0] if args else time.strftime('live_%H%M%S')
    rclpy.init()
    n = Live()
    keys = sorted({k for _, c in CONFIGS for k in c})
    orig = n.get(keys)
    print('valores originales:', orig, flush=True)
    # no empezar hasta que el tracker publique el objetivo (ID elegido + seguimiento activo)
    phase('Elige tu ID en la app web y activa el seguimiento.')
    while True:
        n.tgt, n.rec = [], True
        n.spin_for(3.0)
        n.rec = False
        if len(n.tgt) >= 15:
            break
        print('esperando objetivo: elige tu ID en http://localhost:8080 y activa el seguimiento...', flush=True)
    res = []
    try:
        for i, (lab, prm) in enumerate(CONFIGS, 1):
            n.set({**orig, **prm})
            phase(f'Configuración {i} de {len(CONFIGS)}. {SAY}')
            print(f'[{i}/{len(CONFIGS)}] {lab}: {prm or "valores base"}', flush=True)
            n.spin_for(SETTLE)
            n.js, n.tgt, n.rec = [], [], True
            n.spin_for(REC)
            n.rec = False
            if len(n.js) < 100:
                print('   sin datos de articulaciones', flush=True)
                continue
            if len(n.tgt) < 0.5 * REC * 10:
                print(f'   INVÁLIDA: objetivo solo {len(n.tgt) / REC:.1f} Hz (perdido o seguimiento en pausa)', flush=True)
                phase('Se perdió el objetivo. Vuelve frente a la cámara.')
                continue
            m = metrics(n.js, n.tgt)
            res.append((lab, prm, m))
            print('   ' + '  '.join(f'{k} {v:.2f}' for k, v in m.items()), flush=True)
    finally:
        n.set(orig)
        phase('FIN')
        print('valores originales restaurados', flush=True)
    os.makedirs(f'{DIAG}/{name}', exist_ok=True)
    json.dump([{'config': lab, 'params': prm, **m} for lab, prm, m in res],
              open(f'{DIAG}/{name}/live.json', 'w'), indent=1)
    print(f"\n{'combinación':20s} {'vib mm':>7s} {'vib °':>6s} {'muñeca °':>9s} {'error °':>8s} {'arranq/min':>10s} {'J1 °/s':>7s}")
    for lab, _, m in res:
        print(f"{lab:20s} {m['vib_mm']:7.2f} {m['vib_deg']:6.3f} {m['muneca_deg']:9.3f} {m['err_deg']:8.2f} "
              f"{m['arranques_min']:10.1f} {m['vel_j1']:7.1f}")
    labs = sorted({lab for lab, _, _ in res})
    if len(labs) < len(res):                      # repeticiones: promedio ± desviación por grupo
        print('\npromedios (± desviación):')
        for lab in labs:
            ms = [m for l_, _, m in res if l_ == lab]
            print(f'{lab:20s} ' + '  '.join(
                f'{k} {np.nanmean([m[k] for m in ms]):.2f}±{np.nanstd([m[k] for m in ms]):.2f}' for k in ms[0]))


if __name__ == '__main__':
    main()
