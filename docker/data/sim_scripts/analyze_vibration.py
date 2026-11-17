import pickle, numpy as np
from scipy.spatial.transform import Rotation as R
r = pickle.load(open('/home/ubuntu/.ros/lite6_handeye/diag/vib.pkl', 'rb'))
def stats(name, t, X, unit, scale=1.0):
    X = np.asarray(X) * scale; t = np.asarray(t)
    d = np.diff(X, axis=0)
    flips = (np.sign(d[1:]) * np.sign(d[:-1]) < 0).mean(axis=0) if len(d) > 2 else None
    print(f'{name}: n={len(X)} {1/np.median(np.diff(t)):.1f} Hz | std paso-a-paso {np.round(np.std(d, axis=0), 3)} {unit}'
          f' | rango {np.round(np.ptp(X, axis=0), 2)} | inversiones de signo {np.round(flips, 2)}')
tt = [x[0] for x in r['tgt']]; stats('objetivo xyz (mm)', tt, [x[1:] for x in r['tgt']], 'mm', 1000)
te = [x[0] for x in r['eef']]; P = np.array([x[1:4] for x in r['eef']]); Q = np.array([x[4:] for x in r['eef']])
stats('TCP deseado xyz (mm)', te, P, 'mm', 1000)
stats('TCP deseado rpy (deg)', te, np.unwrap(R.from_quat(Q).as_euler('xyz'), axis=0), 'deg', 180/np.pi)
tc = [x[0] for x in r['cmd']]; C = np.array([x[1][0][1] for x in r['cmd']])
stats('comando q[0] (deg)', tc, C, 'deg', 180/np.pi)
print('puntos por comando:', sorted(set(len(x[1]) for x in r['cmd'])), ' t_from_start 1er punto:', np.round(np.median([x[1][0][0] for x in r['cmd']]), 3))
js = r['js']; names = js[0][1]; idx = [names.index(f'joint{i}') for i in range(1, 7)]
tj = [x[0] for x in js]; J = np.array([[x[2][i] for i in idx] for x in js]); V = np.array([[x[3][i] if x[3] else 0 for i in idx] for x in js])
stats('real q (deg)', tj, J, 'deg', 180/np.pi)
print('real |v| media (deg/s):', np.round(np.degrees(np.abs(V)).mean(axis=0), 1), ' max:', np.round(np.degrees(np.abs(V)).max(axis=0), 1))
print('J5 real (deg) min/max:', np.round(np.degrees(J[:, 4]).min(), 1), np.round(np.degrees(J[:, 4]).max(), 1))
# espectro de la velocidad real: frecuencia dominante por articulación
dt = np.median(np.diff(tj))
for k in range(6):
    v = np.gradient(J[:, k], tj); v = v - v.mean()
    f = np.fft.rfftfreq(len(v), dt); a = np.abs(np.fft.rfft(v)); a[0] = 0
    print(f'J{k+1} pico vel {f[a.argmax()]:.2f} Hz, amplitud vel rms {np.degrees(v.std()):.1f} deg/s')
