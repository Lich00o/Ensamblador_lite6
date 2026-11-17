#!/usr/bin/env python3
"""Métricas de vibración del robot real (misma definición que tuning_replay.py) sobre una
grabación de rec_vibration.py. Uso: python3 analyze_real.py diag/vib_x.pkl [...]"""
import pickle, sys
import numpy as np
from lite6_handeye import calib_io
from lite6_handeye.person_tracker import lite6_fk

X = calib_io.tf_dict_to_T(calib_io.load_result('/home/ubuntu/.ros/lite6_handeye/lite6_zed_calib.yaml')['eef_to_optical'])


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1, 1))))


def movavg(x, n):
    k = np.ones(n)
    num = np.apply_along_axis(lambda c: np.convolve(c, k, 'same'), 0, x)
    return num / np.convolve(np.ones(len(x)), k, 'same')[:, None]


for f in sys.argv[1:]:
    r = pickle.load(open(f, 'rb'))
    js = r['js']
    idx = [js[0][1].index(f'joint{i}') for i in range(1, 7)]
    t = np.array([x[0] for x in js])
    q = np.array([[x[2][i] for i in idx] for x in js])
    v = np.array([[x[3][i] for i in idx] for x in js]) if js[0][3] else np.gradient(q, t, axis=0)
    # remuestreo uniforme a 50 Hz (como la simulación)
    tu = np.arange(t[0], t[-1], 0.02)
    qu = np.stack([np.interp(tu, t, q[:, k]) for k in range(6)], 1)
    T = np.array([lite6_fk(qq) for qq in qu])
    pos, C = T[:, :3, 3], T @ X
    axis, cpos = C[:, :3, 2], C[:, :3, 3]
    jit_p = np.linalg.norm(pos - movavg(pos, 51), axis=1) * 1000
    ax_s = movavg(axis, 51)
    jit_a = np.array([ang(a, b) for a, b in zip(axis, ax_s)])
    # objetivo (salida del Kalman del nodo) suavizado 0.5 s como "verdad"
    tt = np.array([x[0] for x in r['tgt']]) if r['tgt'] else None
    err = np.nan
    if tt is not None and len(tt) > 10:
        # tgt usa reloj monotónico de recepción: alinear por duración (misma ventana de 20 s)
        P = np.array([x[1:] for x in r['tgt']])
        tt = tt - tt[0] + tu[0] + (tu[-1] - tu[0] - (tt[-1] - tt[0])) / 2
        tr = np.stack([np.interp(tu, tt, movavg(P, 15)[:, k]) for k in range(3)], 1)
        err = np.mean([ang(a, g - c) for a, g, c in zip(axis, tr, cpos)])
    moving = np.r_[False, np.max(np.abs(np.diff(qu, axis=0)), axis=1) > 1e-5]
    dur = tu[-1] - tu[0]
    starts = np.sum(moving[1:] & ~moving[:-1]) / dur * 60
    travel = np.degrees(np.abs(np.diff(qu, axis=0))).sum() / dur
    print(f'{f.split("/")[-1]}: vib TCP {np.sqrt(np.mean(jit_p**2)):.2f} mm | vib eje {np.sqrt(np.mean(jit_a**2)):.3f}° | '
          f'error {err:.2f}° | arranques {starts:.1f}/min | recorrido {travel:.2f} °/s | '
          f'vel pico (°/s) {np.round(np.degrees(np.abs(v)).max(0), 1)}')
    # comandado vs real: velocidad pico de la referencia
    if r['cmd']:
        cv = np.array([x[1][0][2] for x in r['cmd']])
        print(f'   vel comandada pico (°/s) {np.round(np.degrees(np.abs(cv)).max(0), 1)}')
