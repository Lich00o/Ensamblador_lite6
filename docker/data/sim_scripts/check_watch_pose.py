"""Revisa la postura actual como pose de vigilancia del person_tracker: límites, geometría
de la cámara (imagen derecha, altura, radio) y continuidad de la IK al seguir desde ahí.
Imprime los valores para pt_real.sh (watch_joints_deg, cam_radius_nom, cam_z_nom)."""
import math
import time

import numpy as np
import rclpy
from sensor_msgs.msg import JointState

from lite6_handeye import calib_io, person_tracker as pt

rclpy.init()
n = rclpy.create_node('check_watch_pose')
js = []
for topic in ('/joint_states', '/ufactory/joint_states'):     # el 2.º publica también en mode 0
    n.create_subscription(JointState, topic, lambda m: js.append(dict(zip(m.name, m.position))), 10)
t0 = time.time()
while not js and time.time() - t0 < 5:
    rclpy.spin_once(n, timeout_sec=0.2)
v = np.degrees([js[-1][f'joint{i}'] for i in range(1, 7)])
w = np.round(v)
q = np.radians(w)
print('ángulos (deg)  ', [round(float(x), 2) for x in v])
print('redondeados    ', w.astype(int).tolist())
lim = np.array([[-178.2, 178.2], [-150, 150], [-3.5, 178.2], [-178.2, 178.2], [-124, 124], [-178.2, 178.2]])
print('margen límites ', np.round(np.minimum(w - lim[:, 0], lim[:, 1] - w)).astype(int).tolist())
X = calib_io.tf_dict_to_T(calib_io.load_result(
    '/home/ubuntu/.ros/lite6_handeye/lite6_zed_calib.yaml')['eef_to_optical'])
C = pt.lite6_fk(q) @ X
p, z, y = C[:3, 3], C[:3, 2], C[:3, 1]
r = float(np.hypot(p[0], p[1]))
az = math.degrees(math.atan2(p[1], p[0]))
pitch = math.degrees(math.asin(np.clip(z[2], -1, 1)))
upright = y[2] < 0
print(f'cámara: radio {r:.3f} m  altura {p[2]:.3f} m  acimut {az:.1f}°  pitch {pitch:+.1f}° (+ = arriba)'
      f'  imagen {"DERECHA" if upright else "INVERTIDA"} (y_opt.z {y[2]:+.2f})')
kw = dict(cam_radius_nom=r, cam_z_nom=float(p[2]), max_azimuth_deg=170,
          cam_radius_range=(max(0.08, r - 0.03), r + 0.05), cam_z_range=(p[2] - 0.08, p[2] + 0.10))
for a, h, dist in ((0, 0.0, 1.5), (30, 0.0, 1.5), (-30, 0.0, 1.5), (90, 0.0, 1.5), (0, 0.6, 1.2), (0, -0.4, 1.0),
                   (150, 0.3, 2.0)):
    ang = math.radians(az + a)
    tgt = np.array([dist * math.cos(ang), dist * math.sin(ang), p[2] + h])
    Td, _ = pt.compute_tracking_pose_6dof(tgt, X, az_ref=math.radians(az), **kw)
    qs, ok = pt.lite6_ik_6dof(Td, q)
    d = np.degrees(qs - q)
    print(f'  persona a {a:+4d}°, {h:+.1f} m de altura, {dist} m: IK {ok}  salto máx {np.max(np.abs(d)):5.1f}°'
          f'  muñeca {np.max(np.abs(d[[3, 5]])):5.1f}°')
rr, zr = kw['cam_radius_range'], kw['cam_z_range']
print(f'\nwatch_joints_deg:=[{", ".join(f"{x:.1f}" for x in w)}]  cam_radius_nom:={r:.2f}  cam_z_nom:={p[2]:.2f}'
      f'  cam_radius_range_m:=[{rr[0]:.2f}, {rr[1]:.2f}]  cam_z_range_m:=[{zr[0]:.2f}, {zr[1]:.2f}]')
