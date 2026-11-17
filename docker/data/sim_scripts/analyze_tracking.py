#!/usr/bin/env python3
"""Análisis completo de una grabación de rec_tracking.py (fluidez del seguimiento)."""
import json, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from lite6_handeye import calib_io
from lite6_handeye.person_tracker import lite6_fk
X = calib_io.tf_dict_to_T(calib_io.load_result('/home/ubuntu/.ros/lite6_handeye/lite6_zed_calib.yaml')['eef_to_optical'])
R = pickle.load(open(sys.argv[1], 'rb'))
D = np.degrees
def hp(x, n):  # pasa-altos: x - media móvil centrada (n muestras)
    k = np.ones(n); num = np.apply_along_axis(lambda c: np.convolve(c, k, 'same'), 0, x)
    return x - num / np.convolve(np.ones(len(x)), k, 'same')[:, None]
def xcorr_delay(a, b, dt, maxlag):  # retardo de b respecto de a (s), b(t) ~ a(t - d)
    a = (a - a.mean()) / (a.std() + 1e-12); b = (b - b.mean()) / (b.std() + 1e-12)
    L = int(maxlag / dt); best = max(range(0, L + 1), key=lambda k: np.dot(a[:len(a) - k], b[k:]) / (len(a) - k))
    c = np.dot(a[:len(a) - best], b[best:]) / (len(a) - best)
    return best * dt, c

# ---- articulaciones reales (150 Hz) remuestreadas a 100 Hz
tj = np.array([x[0] for x in R['js']]); qj = np.array([x[1] for x in R['js']])
t0 = tj[0]; dt = 0.01; tu = np.arange(tj[0], tj[-1], dt)
q = np.stack([np.interp(tu, tj, qj[:, k]) for k in range(6)], 1)
v = np.gradient(q, dt, axis=0); a = np.gradient(v, dt, axis=0); jk = np.gradient(a, dt, axis=0)
T = np.array([lite6_fk(qq) @ X for qq in q]); cpos, cax = T[:, :3, 3], T[:, :3, 2]
print(f'== duración {tu[-1]-tu[0]:.1f} s | joints {len(tj)/(tj[-1]-tj[0]):.0f} Hz | objetivo {len(R["tgt"])/(tj[-1]-tj[0]):.1f} Hz | comandos {len(R["cmd"])/(tj[-1]-tj[0]):.1f} Hz')

# ---- objetivo (salida del Kalman) en base
tt = np.array([x[0] for x in R['tgt']]); P = np.array([x[2:5] for x in R['tgt']])
lat_pub = np.median(np.array([x[1] - x[0] for x in R['tgt']]))
print(f'latencia stamp imagen -> publicación objetivo: mediana {lat_pub*1000:.0f} ms')
gaps = np.diff(tt); print(f'objetivo: huecos > 0.3 s: {np.sum(gaps > 0.3)} (máx {gaps.max():.2f} s)')
dP = np.linalg.norm(np.diff(P, axis=0), axis=1); print(f'objetivo: saltos > 0.2 m entre muestras: {np.sum(dP > 0.2)}')
# rumbo y distancia del objetivo vistos desde la cámara real en ese instante
ci = np.clip(np.searchsorted(tu, tt), 0, len(tu) - 1)
rel = P - cpos[ci]; dist = np.linalg.norm(rel, axis=1)
az_t = np.unwrap(np.arctan2(P[:, 1], P[:, 0])); el_t = np.arctan2(rel[:, 2], np.linalg.norm(rel[:, :2], axis=1))
nt = max(3, int(round(1.0 / np.median(gaps))) | 1)
print(f'objetivo: distancia {dist.min():.2f}–{dist.max():.2f} m | ruido (>1 Hz) posición {np.linalg.norm(hp(P, nt), axis=1).std()*1000:.0f} mm, '
      f'rumbo {D(hp(az_t[:, None], nt)).std():.2f}°, elevación {D(hp(el_t[:, None], nt)).std():.2f}°')
print(f'objetivo: ruido por eje x/y/z (mm): {np.round(hp(P, nt).std(0)*1000, 0)}  (radial ≈ profundidad estimada por bbox)')

# ---- pose deseada del TCP
E = np.array([x[2:5] for x in R['eef']]); Qe = np.array([x[5:9] for x in R['eef']])
rpy = np.unwrap(Rot.from_quat(Qe).as_euler('xyz'), axis=0)
print(f'pose deseada: ruido (>1 Hz) posición {np.linalg.norm(hp(E, nt), axis=1).std()*1000:.1f} mm, orientación {D(hp(rpy, nt)).std(0).round(2)}° (r,p,y)')

# ---- error de apuntado real
ang = []
for k, ti in enumerate(tt):
    i = ci[k]; ang.append(D(np.arccos(np.clip(cax[i] @ rel[k] / dist[k], -1, 1))))
ang = np.array(ang)
print(f'error de apuntado: mediana {np.median(ang):.1f}°, p90 {np.percentile(ang, 90):.1f}°, máx {ang.max():.1f}° | '
      f'<5° (zona muerta) {np.mean(ang < 5)*100:.0f}% del tiempo')

# ---- retardo del lazo: rumbo de la persona vs rumbo del eje óptico
az_c = np.unwrap(np.arctan2(cax[:, 1], cax[:, 0])); az_ti = np.interp(tu, tt, az_t)
d_, c_ = xcorr_delay(np.gradient(az_ti), np.gradient(az_c), dt, 3.0)
print(f'retardo persona -> cámara (rumbo): {d_*1000:.0f} ms (correlación {c_:.2f})')

# ---- comandos: referencia del suavizador vs robot real
tc = np.array([x[0] for x in R['cmd']]); la = np.array([x[1][0][0] for x in R['cmd']])
qc = np.array([x[1][0][1] for x in R['cmd']]); vc = np.array([x[1][0][2] for x in R['cmd']])
cg = np.diff(tc); bursts = np.sum(cg > 0.1) + 1
print(f'comandos: {bursts} ráfagas, activo {np.sum(cg[cg < 0.1])/(tu[-1]-tu[0])*100:.0f}% del tiempo, lookahead {np.median(la):.2f} s')
qr = np.stack([np.interp(tc + la, tu, q[:, k]) for k in range(6)], 1)   # real en el instante objetivo del punto
err = D(qr - qc)
print('seguimiento referencia->real (°): rms ' + ' '.join(f'J{k+1} {np.sqrt(np.mean(err[:, k]**2)):.2f}' for k in range(6)))
for k in (0, 2):
    vr = np.interp(tc, tu, v[:, k]); d2, c2 = xcorr_delay(vc[:, k], vr, np.median(cg[cg < 0.1]), 0.6)
    print(f'  J{k+1}: retardo velocidad comandada -> real {d2*1000:.0f} ms (corr {c2:.2f}), pico comandado {D(np.abs(vc[:, k])).max():.1f}°/s, real {D(np.abs(v[:, k])).max():.1f}°/s')

# ---- suavidad del movimiento real
mv = np.max(np.abs(D(v)), axis=1) > 0.5
starts = np.sum(mv[1:] & ~mv[:-1]); seg = np.diff(np.r_[0, np.flatnonzero(np.diff(mv.astype(int))) + 1, len(mv)])
print(f'movimiento: en marcha {mv.mean()*100:.0f}% | arranques {starts} ({starts/(tu[-1]-tu[0])*60:.0f}/min) | tramo medio {np.median(seg)*dt:.2f} s')
for k in range(6):
    s = np.sign(v[:, k]) * (np.abs(D(v[:, k])) > 0.5); s = s[s != 0]
    rev = np.sum(s[1:] != s[:-1])
    print(f'  J{k+1}: rango {D(np.ptp(q[:, k])):5.1f}° | v pico {D(np.abs(v[:, k])).max():5.1f}°/s | a pico {D(np.abs(a[:, k])).max():6.0f}°/s² | '
          f'jerk rms {D(np.sqrt(np.mean(jk[:, k]**2))):7.0f}°/s³ | inversiones {rev} | osc. >5 Hz {D(hp(q[:, [k]], 21)).std():.3f}°')
st = sorted(set((x[1], x[2], x[3]) for x in R['rs'])); print('estados robot (state, mode, err):', st)
ids = []
for _, s in R['tracks']:
    d = json.loads(s); ids.append(d['target'])
chg = sum(1 for i in range(1, len(ids)) if ids[i] != ids[i - 1]); print(f'objetivo ID: cambios {chg}, valores {sorted(set(map(str, ids)))}')
