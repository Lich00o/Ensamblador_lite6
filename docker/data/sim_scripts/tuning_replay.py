#!/usr/bin/env python3
"""Fase A del afinado: repite una grabación de tuning_record.py por el lazo del person_tracker
con un robot virtual y mide, por fase de la grabación, cuánto vibra y cuánto se atrasa el TCP.

Lazo simulado (igual que person_tracker_node): detección YOLO (una vez, en caché) -> rayo con la
pose real de la cámara -> Kalman -> compute_tracking_pose_6dof -> zona muerta con histéresis ->
IK -> JointSmoother (Ruckig) a stream_rate_hz. El robot virtual sigue la referencia del
suavizador (sin la dinámica del controlador real: eso es la fase B).

Métricas por fase:
  jit_mm   vibración del TCP (link_eef): RMS de la posición menos su media móvil de 1 s
  jit_deg  vibración del eje óptico: RMS del ángulo respecto de su media móvil de 1 s
  err_deg  error de apuntado: ángulo eje óptico -> persona "verdadera" (mediciones suavizadas
           sin retraso, 0.5 s centrado); mezcla retraso y zona muerta
  starts   arranques del brazo por minuto (de quieto a moviéndose)
  travel   recorrido articular (grados/s, suma de las 6): lo que se gasta en vibrar

Uso (en lite6_gpu):
  python3 tuning_replay.py tuning1                 # barrido de un parámetro a la vez
  python3 tuning_replay.py tuning1 --set kf_accel_std=0.15 cam_z_gain=0   # una combinación
"""
import argparse
import glob
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

from lite6_handeye import byte_tracker, calib_io, person_tracker
from lite6_handeye.motion_smoother import JointSmoother

DIAG = os.path.expanduser('~/.ros/lite6_handeye/diag')
CONFIG = '/ws/install/lite6_handeye/share/lite6_handeye/config/lite6_handeye.yaml'
MODEL = os.path.expanduser('~/.ros/lite6_handeye/models/yolo26n-pose.onnx')

# configuración base = config + lo que fija pt_real.sh + el comando de prueba actual
PT_REAL = {'watch_joints_deg': [-5.0, -36.0, 25.0, -11.0, -29.0, -80.0], 'cam_radius_nom': 0.22,
           'cam_z_nom': 0.30, 'cam_radius_range_m': [0.18, 0.28], 'cam_z_range_m': [0.22, 0.40],
           'max_joint_vel_rad_s': 0.5, 'deadband_enter_deg': 5.0, 'joint_deadband_rad': 0.03}

SWEEP = {
    'kf_accel_std': [0.1, 0.2, 0.3, 0.6],
    'kf_meas_lat_std': [0.03, 0.05, 0.1],
    'kf_meas_depth_std': [0.15, 0.3, 0.6],
    'cam_z_gain': [0.0, 0.14, 0.28],
    'cam_radius_gain': [0.0, 0.05],
    'use_fixed_distance': [False, True],
    'deadband_enter_deg': [3.0, 5.0, 8.0],
    'deadband_exit_deg': [1.0, 2.0, 3.5],
    'joint_deadband_rad': [0.005, 0.02, 0.03, 0.05],
    'control_period': [0.05, 0.08, 0.15],
    'max_joint_jerk_rad_s3': [3.0, 8.0, 20.0],
    'max_joint_acc_rad_s2': [0.8, 1.5, 3.0],
}


def base_params():
    p = yaml.safe_load(open(CONFIG))['person_tracker']['ros__parameters']
    p.update(PT_REAL)
    p.setdefault('default_person_distance', 2.0)
    p['default_person_distance'] = 2.0
    p.setdefault('use_fixed_distance', False)
    return p


def detect(name):
    """Detecciones de todas las imágenes (caché en dets.pkl)."""
    d = f'{DIAG}/{name}'
    cache = f'{d}/dets.pkl'
    if os.path.exists(cache):
        return pickle.load(open(cache, 'rb'))
    import cv2
    meta = json.load(open(f'{d}/meta.json'))
    K = np.array(meta['camera_info']['K']).reshape(3, 3)
    det = person_tracker.PersonDetector(onnx_model=MODEL, ref_torso_height_m=0.52, min_score=0.45)
    files = sorted(glob.glob(f'{d}/img/*.jpg'), key=lambda f: int(os.path.basename(f)[:-4]))
    out = []
    for i, f in enumerate(files):
        dets = det.detect_all(cv2.imread(f), K, 0.15)
        out.append((int(os.path.basename(f)[:-4]) * 1e-9,
                    [(d_.bbox, d_.track_uv, d_.est_distance, d_.score) for d_ in dets]))
        if i % 500 == 0:
            print(f'  detección {i}/{len(files)}', flush=True)
    pickle.dump(out, open(cache, 'wb'))
    return out


def load(name):
    d = f'{DIAG}/{name}'
    meta = json.load(open(f'{d}/meta.json'))
    J = np.load(f'{d}/joints.npz')
    ph_file = f'{d}/phases.json' if os.path.exists(f'{d}/phases.json') else f'{DIAG}/{name}.phases.json'
    phases = json.load(open(ph_file))
    X = calib_io.tf_dict_to_T(calib_io.load_result(meta['calibration_file'])['eef_to_optical'])
    return dict(K=np.array(meta['camera_info']['K']).reshape(3, 3), q_rec=np.median(J['q'], axis=0),
                X=X, phases=phases, dets=detect(name))


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1, 1))))


def simulate(data, p):
    K, X, q_rec = data['K'], data['X'], data['q_rec']
    T_eef_rec = person_tracker.lite6_fk(q_rec)                # robot real quieto durante la grabación
    cam_rec = (T_eef_rec @ X)[:3, 3]
    kf = person_tracker.TargetKalman(p['kf_accel_std'], p['kf_meas_lat_std'], p['kf_meas_depth_std'],
                                     p['kf_gate_sigma'], p['kf_max_rejects'])
    bt = byte_tracker.ByteTracker(high_thresh=p['track_high_thresh'], low_thresh=p['track_low_thresh'],
                                  new_track_thresh=p['track_high_thresh'] + 0.1, track_buffer=60, reid=False)
    dt = 1.0 / p['stream_rate_hz']
    sm = JointSmoother(6, dt, p['max_joint_vel_rad_s'], p['max_joint_acc_rad_s2'], p['max_joint_jerk_rad_s3'])
    q_watch = np.deg2rad(p['watch_joints_deg'])
    T_cw = person_tracker.lite6_fk(q_watch) @ X
    az_watch = float(np.arctan2(T_cw[1, 3], T_cw[0, 3]))
    q_v = q_rec.copy()                                        # robot virtual: parte donde estaba el real
    wrist_lim = np.deg2rad(p['max_wrist_jump_deg'])
    centering, last_cmd_t, last_cmd_q, target_id = False, -1e9, None, None
    t_tick = data['dets'][0][0]
    log_t, log_q, meas_t, meas_p = [], [], [], []
    for t_img, dets in data['dets']:
        while t_tick < t_img:                                 # suavizador a stream_rate_hz
            if sm.active and not sm.settled():
                q_v, _, _, _ = sm.step(lookahead=p['stream_lookahead_s'])
            log_t.append(t_tick)
            log_q.append(q_v.copy())
            t_tick += dt
        objs = [person_tracker.PersonDetection(bbox=b, track_uv=uv, head_uv=uv, est_distance=dd, score=s,
                                               method='yolo-pose') for b, uv, dd, s in dets]
        tracks = bt.update(objs)
        if target_id is None or bt.get(target_id) is None:
            t_ = byte_tracker.select_nearest(tracks)
            target_id = None if t_ is None else t_.id
        tg = bt.get(target_id)
        if tg is None or not tg.visible:
            continue
        det = tg.det
        dist = p['default_person_distance'] if p['use_fixed_distance'] else det.est_distance
        p_meas = person_tracker.ray_to_target_3d(det.track_uv[0], det.track_uv[1], K, T_eef_rec, X, dist)
        meas_t.append(t_img)
        meas_p.append(p_meas)
        kf.update(p_meas, t_img, ray_dir=p_meas - cam_rec)
        target = kf.position
        T_cam_v = person_tracker.lite6_fk(q_v) @ X
        az_now = float(np.arctan2(T_cam_v[1, 3], T_cam_v[0, 3]))
        T_eef_des, T_cam_des = person_tracker.compute_tracking_pose_6dof(
            target, X, cam_radius_nom=p['cam_radius_nom'], cam_radius_gain=p['cam_radius_gain'],
            cam_z_nom=p['cam_z_nom'], cam_z_gain=p['cam_z_gain'], max_azimuth_deg=p['max_azimuth_deg'],
            cam_radius_range=tuple(p['cam_radius_range_m']), cam_z_range=tuple(p['cam_z_range_m']),
            az_ref=az_now)
        a = ang(T_cam_v[:3, 2], target - T_cam_v[:3, 3])
        if centering and a < p['deadband_exit_deg']:
            centering = False
        elif not centering and a > p['deadband_enter_deg']:
            centering = True
        if centering and t_img - last_cmd_t >= p['control_period']:
            seed = last_cmd_q if last_cmd_q is not None else q_v
            q_sol, ok = person_tracker.lite6_ik_6dof(T_eef_des, seed)
            if not ok or np.max(np.abs((q_sol - q_v)[[3, 5]])) > wrist_lim:
                az_des = float(np.arctan2(T_cam_des[1, 3], T_cam_des[0, 3]))
                q_rot = q_watch.copy()
                q_rot[0] += az_des - az_watch
                q_rot[0] = q_v[0] + np.arctan2(np.sin(q_rot[0] - q_v[0]), np.cos(q_rot[0] - q_v[0]))
                q2, ok2 = person_tracker.lite6_ik_6dof(T_eef_des, q_rot)
                if ok2 and (not ok or np.max(np.abs((q2 - q_v)[[3, 5]])) < np.max(np.abs((q_sol - q_v)[[3, 5]]))):
                    q_sol, ok = q2, ok2
            if ok and np.max(np.abs((q_sol - q_v)[[3, 5]])) > wrist_lim:
                last_cmd_q = None
            elif ok and np.max(np.abs(q_sol - q_v)) >= p['joint_deadband_rad']:
                if not sm.active:
                    sm.reset(q_v)
                sm.set_goal(q_sol)
                last_cmd_q, last_cmd_t = q_sol.copy(), t_img
    return np.array(log_t), np.array(log_q), np.array(meas_t), np.array(meas_p)


def movavg(x, n):
    """Media móvil centrada (n impar) por columna, bordes con la ventana disponible."""
    k = np.ones(n)
    num = np.apply_along_axis(lambda c: np.convolve(c, k, 'same'), 0, x)
    den = np.convolve(np.ones(len(x)), k, 'same')[:, None]
    return num / den


def metrics(data, p, sim):
    t, q, mt, mp = sim
    X = data['X']
    T = np.array([person_tracker.lite6_fk(qq) for qq in q])
    pos = T[:, :3, 3]
    C = T @ X
    axis, cpos = C[:, :3, 2], C[:, :3, 3]
    n1 = int(round(p['stream_rate_hz'])) | 1                   # 1 s
    jit_p = np.linalg.norm(pos - movavg(pos, n1), axis=1) * 1000
    ax_s = movavg(axis, n1)
    jit_a = np.array([ang(a, b) for a, b in zip(axis, ax_s)])
    # persona "verdadera": mediciones crudas suavizadas 0.5 s centrado, interpoladas a t
    if len(mt) > 10:
        nm = max(3, int(0.5 / np.median(np.diff(mt))) | 1)
        truth = movavg(mp, nm)
        tr = np.stack([np.interp(t, mt, truth[:, k]) for k in range(3)], 1)
        err = np.array([ang(a, g - c) for a, g, c in zip(axis, tr, cpos)])
        seen = np.interp(t, mt, np.ones(len(mt)), left=0, right=0) > 0
    else:
        err, seen = np.full(len(t), np.nan), np.zeros(len(t), bool)
    moving = np.r_[False, np.max(np.abs(np.diff(q, axis=0)), axis=1) > 1e-5]
    dq = np.r_[0.0, np.degrees(np.abs(np.diff(q, axis=0))).sum(axis=1)]
    out = {}
    for ph in data['phases']:
        if ph['fase'] == 'PREPARACION':
            continue
        m = (t >= ph['t0'] + 1.0) & (t < ph['t1'])            # 1 s de margen al cambiar de fase
        dur = max(m.sum() / p['stream_rate_hz'], 1e-3)
        mv = moving[m]
        out[ph['fase']] = dict(
            jit_mm=float(np.sqrt(np.mean(jit_p[m] ** 2))), jit_deg=float(np.sqrt(np.mean(jit_a[m] ** 2))),
            err_deg=float(np.nanmean(err[m & seen])) if (m & seen).any() else float('nan'),
            starts=float(np.sum(mv[1:] & ~mv[:-1]) / dur * 60), travel=float(dq[m].sum() / dur))
    return out


def run_one(args):
    data, p, label = args
    return label, metrics(data, p, simulate(data, p))


def score(r):
    """Resumen: vibración (quieto + lateral) y error de apuntado (lateral/profundidad)."""
    q = [r[k] for k in ('QUIETO', 'QUIETO CERCA') if k in r]
    mv = [r[k] for k in ('LATERAL LENTO', 'PROFUNDIDAD', 'LATERAL RAPIDO') if k in r]
    return dict(jit_quieto_mm=np.mean([x['jit_mm'] for x in q]), jit_mov_mm=np.mean([x['jit_mm'] for x in mv]),
                jit_mov_deg=np.mean([x['jit_deg'] for x in mv]), err_deg=np.nanmean([x['err_deg'] for x in mv]),
                starts_min=np.mean([x['starts'] for x in q + mv]), travel=np.mean([x['travel'] for x in q + mv]))


def parse_set(items):
    out = {}
    for it in items:
        k, v = it.split('=', 1)
        out[k] = v.lower() == 'true' if v.lower() in ('true', 'false') else float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--set', nargs='*', default=[], help='param=valor sobre la base')
    ap.add_argument('--sweep', nargs='*', default=None, help='parámetros a barrer (por defecto todos)')
    ap.add_argument('--grid', nargs='*', default=None, help='k=v1,v2 ... : todas las combinaciones')
    ap.add_argument('--top', type=int, default=0, help='con --grid: muestra solo las N mejores')
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()
    data = load(a.name)
    base = base_params()
    base.update(parse_set(a.set))
    print(f"detecciones: {sum(1 for _, d in data['dets'] if d)}/{len(data['dets'])} cuadros con persona")
    jobs = [(data, base, 'BASE')]
    if a.grid:
        import itertools
        keys = [g.split('=')[0] for g in a.grid]
        vals = [[parse_set([f'{k}={v}'])[k] for v in g.split('=')[1].split(',')] for k, g in zip(keys, a.grid)]
        for combo in itertools.product(*vals):
            jobs.append((data, {**base, **dict(zip(keys, combo))},
                         ' '.join(f'{k.replace("max_joint_", "").replace("_rad_s3", "").replace("_rad_s2", "")}={v:g}'
                                  if not isinstance(v, bool) else f'{k}={v}' for k, v in zip(keys, combo))))
    elif not a.set or a.sweep is not None:
        for k in (a.sweep or SWEEP):
            for v in SWEEP[k]:
                if base.get(k) != v:
                    jobs.append((data, {**base, k: v}, f'{k}={v}'))
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run_one, jobs))
    rows = [(lab, score(r), r) for lab, r in res]
    if a.grid and a.top:
        # orden: vibración (quieto + movimiento) con penalización por error de apuntado sobre la base
        e0 = rows[0][1]['err_deg']
        key = lambda x: x[1]['jit_quieto_mm'] + x[1]['jit_mov_mm'] + 2.0 * max(0.0, x[1]['err_deg'] - e0)  # noqa: E731
        rows = rows[:1] + sorted(rows[1:], key=key)[:a.top]
    hdr = f"{'combinación':62s} {'vib quieto mm':>13s} {'vib mov mm':>10s} {'vib mov °':>9s} {'error °':>8s} {'arranq/min':>10s} {'recorr °/s':>10s}"
    print(hdr)
    print('-' * len(hdr))
    for lab, s, _ in rows:
        print(f"{lab:62s} {s['jit_quieto_mm']:13.2f} {s['jit_mov_mm']:10.2f} {s['jit_mov_deg']:9.3f} "
              f"{s['err_deg']:8.2f} {s['starts_min']:10.1f} {s['travel']:10.2f}")
    if len(rows) == 1:
        print('\npor fase:')
        for ph, m in rows[0][2].items():
            print(f'  {ph:15s} ' + '  '.join(f'{k} {v:.2f}' for k, v in m.items()))
    json.dump({lab: {'score': s, 'fases': r} for lab, s, r in rows},
              open(f'{DIAG}/{a.name}/replay_{"_".join(a.set) or "sweep"}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
