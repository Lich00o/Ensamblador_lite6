"""Captura una nube de /sim_depth/points y resume qué hay cerca de cada objeto (frame link_base)."""
import sys, time, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
sys.path.insert(0, "/ws/src/lite6_handeye")
from lite6_handeye.transforms import T_from_xyz_quat

rclpy.init(); n = Node("cloud_probe", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
buf = Buffer(); TransformListener(buf, n); got = {}
n.create_subscription(PointCloud2, "/sim_depth/points", lambda m: got.setdefault("m", m), 1)
t0 = time.time()
while "m" not in got and time.time() - t0 < 20: rclpy.spin_once(n, timeout_sec=0.1)
for _ in range(30): rclpy.spin_once(n, timeout_sec=0.05)
m = got["m"]
P = pc2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True)
P = P[np.isfinite(P).all(1)]
r = np.linalg.norm(P, axis=1)
print("rango sin filtro: percentiles 0/5/50/95/100", np.round(np.percentile(r, [0, 5, 50, 95, 100]), 3))
P = P[r >= 0.6]  # mínimo de la Astra Pro Plus
if not len(P): sys.exit("sin puntos >= 0.6 m")
tf = buf.lookup_transform("link_base", "sim_camera_link", rclpy.time.Time()).transform
T = T_from_xyz_quat([tf.translation.x, tf.translation.y, tf.translation.z],
                    [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])
Q = P @ T[:3, :3].T + T[:3, 3]
print(f"puntos {len(P)}  rango cámara {np.linalg.norm(P,axis=1).min():.3f}-{np.linalg.norm(P,axis=1).max():.3f} m")
print("z>0.01 (sobre la mesa):", int((Q[:, 2] > 0.01).sum()))
for name, top in {"box_a": [0.24, 0.20, 0.08], "cyl_b": [0.33, -0.20, 0.06]}.items():
    near = Q[np.linalg.norm(Q[:, :2] - top[:2], axis=1) < 0.06]
    print(name, "puntos a <6 cm en xy:", len(near), " z máx", round(float(near[:, 2].max()), 4) if len(near) else None)
