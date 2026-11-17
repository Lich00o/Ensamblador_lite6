"""Mapa de la mesa observable con la profundidad de la Astra (>= DMIN m) usando el soporte real.
Para cada punto de la mesa (a la altura Z del objeto), busca una vista alcanzable sin colisiones."""
import sys, numpy as np, rclpy, yaml
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
sys.path.insert(0, "/ws/src/lite6_handeye")
from lite6_handeye import pose_sampler
from lite6_handeye.transforms import T_from_xyz_rpy, fill_ros_pose
CFG = yaml.safe_load(open("/ws/src/lite6_handeye/config/lite6_handeye.yaml"))
X0 = T_from_xyz_rpy(*np.split(np.array(CFG["handeye_auto_calibrator"]["ros__parameters"]["x_prior_xyz_rpy"]), 2))
DMIN, Z = 0.62, 0.06
rclpy.init(); n = Node("observable_map"); ik = n.create_client(GetPositionIK, "/compute_ik"); ik.wait_for_service()
def ok(T):
    r = GetPositionIK.Request(); q = r.ik_request; q.group_name = "lite6"; q.ik_link_name = "link_eef"
    q.avoid_collisions = True; q.timeout.nanosec = 150_000_000; q.pose_stamped.header.frame_id = "link_base"
    fill_ros_pose(q.pose_stamped.pose, T)
    q.robot_state.joint_state.name = [f"joint{i}" for i in range(1, 7)]; q.robot_state.joint_state.position = [0, 0.2, 1.0, 0, 0.8, 0]
    f = ik.call_async(r); rclpy.spin_until_future_complete(n, f, timeout_sec=5)
    return f.result() is not None and f.result().error_code.val == 1
xs = np.arange(0.15, 0.46, 0.05); ys = np.arange(0.30, -0.31, -0.10)
print("y\\x   " + " ".join(f"{x:5.2f}" for x in xs), flush=True)
for y in ys:
    row = []
    for x in xs:
        c = [(e, t) for e, t in pose_sampler.observation_viewpoints(np.array([x, y, Z]), X0, dist_range=(DMIN, DMIN + 0.12),
                                                                     elev_range_deg=(15, 75)) if t[2, 3] <= 0.62]
        hit = None
        for e, t in c[::2]:
            if ok(e):
                ray = np.array([x, y, Z]) - t[:3, 3]; hit = np.degrees(np.arcsin(-ray[2] / np.linalg.norm(ray))); break
        row.append("  -- " if hit is None else f"{hit:4.0f}°")
    print(f"{y:+.2f} " + " ".join(row), flush=True)
