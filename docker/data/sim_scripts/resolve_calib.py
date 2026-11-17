"""Re-resuelve la calibración Hand-Eye con las muestras guardadas: compara métodos,
estabilidad leave-one-out y bootstrap. No mueve el robot."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from lite6_handeye import calib_io, handeye_solver as hs
from lite6_handeye.transforms import T_from_xyz_rpy

base = '/home/ubuntu/.ros/lite6_handeye/'
frames, G, C, T_ro = calib_io.load_samples(base + 'lite6_zed_calib_samples.yaml')
G, C = list(G), list(C)
n = len(G)


def ev(X, idx=None):
    idx = range(n) if idx is None else idx
    P = np.array([(G[i] @ X @ C[i])[:3, 3] for i in idx])
    Rs = R.from_matrix([(G[i] @ X @ C[i])[:3, :3] for i in idx])
    rot = np.degrees((Rs.mean().inv() * Rs).magnitude())
    return np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, 1))) * 1000, np.sqrt(np.mean(rot ** 2))


def dX(X, Y):
    return (np.linalg.norm(X[:3, 3] - Y[:3, 3]) * 1000,
            np.degrees(R.from_matrix(X[:3, :3].T @ Y[:3, :3]).magnitude()))


sol = {}
for m in ('park', 'refine'):
    sol[m] = hs.solve(G, C, method=m)[0] if isinstance(hs.solve(G, C, method=m), tuple) else hs.solve(G, C, method=m)
cvm = {'tsai': cv2.CALIB_HAND_EYE_TSAI, 'horaud': cv2.CALIB_HAND_EYE_HORAUD,
       'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS, 'andreff': cv2.CALIB_HAND_EYE_ANDREFF}
for k, f in cvm.items():
    Rx, tx = cv2.calibrateHandEye([g[:3, :3] for g in G], [g[:3, 3] for g in G],
                                  [c[:3, :3] for c in C], [c[:3, 3] for c in C], method=f)
    X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = tx.ravel()
    sol[k] = X
ref = sol['refine']
cad = T_from_xyz_rpy([0.0125, -0.070, 0.042], [0, 0, np.pi])
print(f'{"método":11s} {"tablero RMS":>14s} {"vs refine":>16s} {"vs CAD":>16s}   t (mm)')
for k, X in sol.items():
    t, r = ev(X)
    a, b = dX(X, ref)
    c, e = dX(X, cad)
    print(f'{k:11s} {t:6.2f}mm {r:5.2f}°  {a:6.2f}mm {b:5.2f}°  {c:6.2f}mm {e:5.2f}°   {np.round(X[:3, 3] * 1000, 1)}')

print('\nleave-one-out (refine): cambio de X al quitar cada muestra')
loo = []
for i in range(n):
    idx = [j for j in range(n) if j != i]
    X = hs.solve([G[j] for j in idx], [C[j] for j in idx], method='refine')
    X = X[0] if isinstance(X, tuple) else X
    a, b = dX(X, ref)
    t, _ = ev(X, idx)
    loo.append((i, a, b, t))
    print(f'  sin m{i:2d}: ΔX {a:5.2f} mm {b:5.2f}°   RMS resto {t:5.2f} mm')

rng = np.random.default_rng(0)
bt = []
for _ in range(200):
    idx = rng.choice(n, 8, replace=False)        # submuestreo 8/11 (con reposición degenera Park)
    try:
        X = hs.solve([G[j] for j in idx], [C[j] for j in idx], method='refine')
    except ValueError:
        continue
    X = X[0] if isinstance(X, tuple) else X
    bt.append(X)
tt = np.array([X[:3, 3] for X in bt]) * 1000
rr = np.array([dX(X, ref)[1] for X in bt])
print(f'\nsubmuestreo 8/{n} ({len(bt)} corridas): σ traslación x,y,z mm', np.round(tt.std(0), 2), '  rot 95%% %.2f°' % np.percentile(rr, 95))
