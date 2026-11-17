"""Calibración de intrínsecos de la Orbbec (RGB) con el ChArUco, sin mover el robot.

Ventana en vivo con la imagen dividida en una grilla 3x3. Cada celda necesita 2 capturas con
el centro del tablero dentro de ella, a menos de 60 cm: una PLANA (<15°) y una INCLINADA (25-50°).
  rojo = falta todo, amarillo = falta una, verde = lista.
La celda sugerida se marca con borde blanco. Los puntos azules son esquinas ya capturadas
(cobertura para la distorsión: conviene llenar también los bordes de la imagen).

La captura es automática cuando el tablero está quieto ~1 s y aporta a una celda incompleta.
Al completar todas las celdas (y >= MIN_FRAMES) calibra con modelo restringido (k1, k2; sin
tangencial ni k3), descarta vistas malas, guarda:
  ~/.ros/lite6_handeye/orbbec_intrinsics.yaml   (lo lee orbbec_uvc al arrancar)
  ~/.ros/lite6_handeye/orbbec_intrinsics_corners.npz   (para re-resolver offline)

Uso (dentro del contenedor): python3 calib_intrinsics.py
Teclas: c = calibrar ya (>= 10 capturas), q = salir sin guardar.
"""
import os
import time

import cv2
import numpy as np
import yaml

from lite6_handeye.charuco import BoardConfig, CharucoPoseEstimator

BASE = os.path.expanduser('~/.ros/lite6_handeye')
SRC = os.path.join(BASE, '.live_cam.jpg')
OUT = os.path.join(BASE, 'orbbec_intrinsics.yaml')
NPZ = os.path.join(BASE, 'orbbec_intrinsics_corners.npz')
MIN_CORNERS = 12
MIN_FRAMES = 20
GRID = 3
FLAT_MAX, TILT_MIN, TILT_MAX = 15.0, 25.0, 50.0
MAX_DIST = 0.60                 # m: lejos, focal y distorsión no se separan
FLAGS = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3

cfg_all = yaml.safe_load(open('/ws/src/lite6_handeye/config/lite6_handeye.yaml'))['/**']['ros__parameters']
cfg = BoardConfig(cfg_all['board.squares_x'], cfg_all['board.squares_y'], cfg_all['board.square_length'],
                  cfg_all['board.marker_length'], cfg_all['board.dictionary'])
est = CharucoPoseEstimator(cfg)


def read_frame():
    try:
        buf = open(SRC, 'rb').read()
    except OSError:
        return None
    if len(buf) < 256 or buf[:2] != b'\xff\xd8' or buf[-2:] != b'\xff\xd9':
        return None
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def classify(corners, ids, K0, w, h):
    """Celda (col, fila) del centro del tablero y tipo 'flat' | 'tilt' | None (con K nominal)."""
    c = corners.mean(0)
    cell = (min(int(GRID * c[0] / w), GRID - 1), min(int(GRID * c[1] / h), GRID - 1))
    ok, rvec, tvec = cv2.solvePnP(est.obj_corners[ids], corners, K0, None)
    if not ok:
        return cell, None, 0.0, 9.9
    tilt = float(np.degrees(np.arccos(abs(cv2.Rodrigues(rvec)[0][2, 2]))))
    kind = 'flat' if tilt < FLAT_MAX else 'tilt' if TILT_MIN <= tilt <= TILT_MAX else None
    return cell, kind, tilt, float(np.linalg.norm(tvec))


def draw_grid(vis, cov, target, w, h, pts):
    over = vis.copy()
    cw, ch = w / GRID, h / GRID
    for (i, j), got in cov.items():
        n = len(got)
        color = (0, 0, 255) if n == 0 else (0, 220, 255) if n == 1 else (0, 200, 0)
        x0, y0 = int(i * cw), int(j * ch)
        cv2.rectangle(over, (x0, y0), (int(x0 + cw), int(y0 + ch)), color, -1)
    cv2.addWeighted(over, 0.22, vis, 0.78, 0, vis)
    for p in pts:
        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (255, 120, 0), -1)
    for (i, j), got in cov.items():
        x0, y0 = int(i * cw), int(j * ch)
        cv2.rectangle(vis, (x0, y0), (int(x0 + cw), int(y0 + ch)), (90, 90, 90), 1)
        txt = ('PLANO ok' if 'flat' in got else 'plano --') + '  ' + \
              ('INCL ok' if 'tilt' in got else 'incl --')
        cv2.putText(vis, txt, (x0 + 10, int(y0 + ch) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    if target is not None:
        (i, j), kind = target
        x0, y0 = int(i * cw), int(j * ch)
        if int(time.time() * 3) % 2:
            cv2.rectangle(vis, (x0 + 3, y0 + 3), (int(x0 + cw) - 3, int(y0 + ch) - 3), (255, 255, 255), 4)
        label = 'tablero AQUI a <60cm ' + ('de frente' if kind == 'flat' else 'inclinado 25-50 grados')
        cv2.putText(vis, label, (x0 + 10, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def next_target(cov):
    order = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2), (1, 0), (1, 2), (0, 1), (2, 1)]
    for kind in ('flat', 'tilt'):
        for c in order:
            if kind not in cov[c]:
                return c, kind
    return None


def calibrate(frames, size, flags):
    def run(sel):
        rms, K, D, _, _, _, _, pv = cv2.aruco.calibrateCameraCharucoExtended(
            [frames[i][0].reshape(-1, 1, 2).astype(np.float32) for i in sel],
            [frames[i][1].reshape(-1, 1).astype(np.int32) for i in sel],
            est.board, size, None, None, flags=flags)
        return rms, K, D, pv.ravel()
    sel = list(range(len(frames)))
    rms, K, D, pv = run(sel)
    bad = [s for s, e in zip(sel, pv) if e > max(2.5 * np.median(pv), 1.0)]
    if bad and len(sel) - len(bad) >= 10:
        sel = [s for s in sel if s not in bad]
        rms, K, D, pv = run(sel)
    return rms, K, D, pv, sel, bad


def main():
    frames, cov, last_cap, prev, K0 = [], {}, 0.0, None, None
    cov = {(i, j): set() for i in range(GRID) for j in range(GRID)}
    pts = []
    cv2.namedWindow('calib_intrinsics', cv2.WINDOW_NORMAL)
    force = False
    while not force:
        target = next_target(cov)
        if target is None and len(frames) >= MIN_FRAMES:
            break
        bgr = read_frame()
        if bgr is None:
            time.sleep(0.03)
            continue
        h, w = bgr.shape[:2]
        if K0 is None:
            f0 = w / (2 * np.tan(np.radians(66.1) / 2))
            K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1.0]])
        corners, ids = est.detect_corners(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        vis = bgr.copy()
        msg = 'muestra el tablero'
        if corners is not None and len(ids) >= MIN_CORNERS:
            cell, kind, tilt, dist = classify(corners, ids, K0, w, h)
            still = (prev is not None and len(prev[1]) == len(ids) and np.all(prev[1] == ids)
                     and np.mean(np.linalg.norm(prev[0] - corners, axis=1)) < 1.0)
            useful = kind is not None and (kind not in cov[cell] or target is None)
            if dist > MAX_DIST:
                msg = f'ACERCA el tablero: {dist * 100:.0f} cm (max {MAX_DIST * 100:.0f})'
            elif kind is None:
                msg = f'inclinacion {tilt:.0f}: usa <15 (plano) o 25-50 (inclinado)'
            elif not useful:
                msg = f'celda ya tiene {"plano" if kind == "flat" else "inclinado"}: ve a otra'
            elif not still:
                msg = f'{"plano" if kind == "flat" else "inclinado"} {tilt:.0f} - deja el tablero quieto'
            elif time.time() - last_cap > 0.8:
                frames.append((corners.copy(), ids.copy()))
                cov[cell].add(kind)
                pts.extend(corners.tolist())
                last_cap = time.time()
                msg = 'CAPTURADO'
            for p in corners:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
            prev = (corners, ids)
        else:
            prev = None
        draw_grid(vis, cov, target, w, h, pts)
        if time.time() - last_cap < 0.4:
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 255, 0), 14)
        done = sum(len(v) == 2 for v in cov.values())
        cv2.putText(vis, f'capturas {len(frames)}  celdas listas {done}/{GRID * GRID}  {msg}',
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow('calib_intrinsics', vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            print('cancelado')
            return
        if k == ord('c') and len(frames) >= 10:
            force = True
    cv2.destroyAllWindows()
    np.savez(NPZ, size=np.array([w, h]), n=len(frames),
             **{f'c{i}': c for i, (c, _) in enumerate(frames)}, **{f'i{i}': d for i, (_, d) in enumerate(frames)})
    rms_f, K_f, D_f, *_ = calibrate(frames, (w, h), 0)
    rms, K, D, pv, sel, bad = calibrate(frames, (w, h), FLAGS)
    out = {
        'image_width': w, 'image_height': h, 'rms_px': float(rms),
        'camera_matrix': [float(v) for v in K.ravel()],
        'distortion_model': 'plumb_bob', 'distortion': [float(v) for v in D.ravel()[:5]],
        'model': 'k1,k2 (tangencial=0, k3=0)',
        'n_views': len(sel), 'discarded_views': len(bad),
        'per_view_rms_px': [round(float(e), 3) for e in pv],
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(OUT, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    hfov = np.degrees(2 * np.arctan(w / (2 * K[0, 0])))
    print(f'{OUT}\n  vistas {len(sel)} (descartadas {len(bad)})  RMS {rms:.3f} px  [modelo k1,k2]\n'
          f'  fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}  HFOV {hfov:.1f}°\n'
          f'  D {np.round(D.ravel()[:5], 4)}\n'
          f'  modelo completo (comparación): RMS {rms_f:.3f} px  fx {K_f[0, 0]:.1f}  D {np.round(D_f.ravel()[:5], 4)}')


if __name__ == '__main__':
    main()
