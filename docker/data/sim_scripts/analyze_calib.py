"""Diagnóstico de una calibración Hand-Eye: consistencia por muestra, geometría de vista y
diversidad de rotaciones. Uso: python3 analyze_calib.py [calib.yaml] [samples.yaml]"""
import itertools
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from lite6_handeye import calib_io

base = '/home/ubuntu/.ros/lite6_handeye/'
cal = sys.argv[1] if len(sys.argv) > 1 else base + 'lite6_zed_calib.yaml'
smp = sys.argv[2] if len(sys.argv) > 2 else base + 'lite6_zed_calib_samples.yaml'
d = yaml.safe_load(open(smp))
X = calib_io.tf_dict_to_T(calib_io.load_result(cal)['eef_to_optical'])
A = [np.array(s['T_base_eef']).reshape(4, 4) for s in d['samples']]
B = [np.array(s['T_cam_board']).reshape(4, 4) for s in d['samples']]
P = np.array([(a @ X @ b)[:3, 3] for a, b in zip(A, B)])     # tablero en base, por muestra
m = P.mean(0)
print('tablero en base (media) mm', np.round(m * 1000, 1))
print('desv. por eje std (mm) x,y,z:', np.round(P.std(0) * 1000, 2))
for i, (b, s) in enumerate(zip(B, d['samples'])):
    dist = np.linalg.norm(b[:3, 3])
    tilt = np.degrees(np.arccos(abs(b[2, 2])))
    j = s['joints']
    print(f'm{i:2d}: desv {np.linalg.norm(P[i] - m) * 1000:4.1f} mm  dist {dist * 1000:3.0f} mm  '
          f'incl {tilt:3.0f}°  J1 {np.degrees(j["joint1"]):5.0f}°  J4 {np.degrees(j["joint4"]):5.0f}°  '
          f'J6 {np.degrees(j["joint6"]):5.0f}°')
ang, ax = [], []
for i, k in itertools.combinations(range(len(A)), 2):
    rv = R.from_matrix((np.linalg.inv(A[i]) @ A[k])[:3, :3]).as_rotvec()
    a = np.linalg.norm(rv)
    ang.append(np.degrees(a))
    ax.append(rv / max(a, 1e-9))
ang, ax = np.array(ang), np.array(ax)
print('ángulo relativo entre pares: min %.0f° mediana %.0f° max %.0f°' % (ang.min(), np.median(ang), ang.max()))
print('dispersión de ejes de rotación (autovalores; ideal ~0.33 cada uno):',
      np.round(np.linalg.eigvalsh(ax.T @ ax / len(ax)), 2))
