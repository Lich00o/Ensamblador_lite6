"""¿Por qué no hay vista para un objeto? IK con y sin colisiones sobre los candidatos."""
import sys, numpy as np, rclpy, yaml
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
sys.path.insert(0, "/ws/src/lite6_handeye")
from lite6_handeye import pose_sampler
from lite6_handeye.transforms import T_from_xyz_rpy, fill_ros_pose
CFG = yaml.safe_load(open("/ws/src/lite6_handeye/config/lite6_handeye.yaml"))
X0 = T_from_xyz_rpy(*np.split(np.array(CFG["handeye_auto_calibrator"]["ros__parameters"]["x_prior_xyz_rpy"]), 2))
A = [a for a in sys.argv[1:] if not a.startswith("--")]
top = np.array([float(v) for v in A[:3]]); dmin = float(A[3]) if len(A) > 3 else 0.65
rclpy.init(); n = Node("view_diag"); ik = n.create_client(GetPositionIK, "/compute_ik"); ik.wait_for_service()
def ok(T, coll):
    r = GetPositionIK.Request(); q = r.ik_request; q.group_name = "lite6"; q.ik_link_name = "link_eef"
    q.avoid_collisions = coll; q.timeout.nanosec = 300_000_000; q.pose_stamped.header.frame_id = "link_base"
    fill_ros_pose(q.pose_stamped.pose, T)
    q.robot_state.joint_state.name = [f"joint{i}" for i in range(1, 7)]; q.robot_state.joint_state.position = [0, 0.2, 1.0, 0, 0.8, 0]
    f = ik.call_async(r); rclpy.spin_until_future_complete(n, f, timeout_sec=5)
    return f.result() is not None and f.result().error_code.val == 1
c = [(e, t) for e, t in pose_sampler.observation_viewpoints(top, X0, dist_range=(dmin, dmin + 0.13), elev_range_deg=(15, 75))
     if "--all" in sys.argv or not (t[2, 3] > 0.60 or np.linalg.norm(e[:2, 3]) > 0.46)]
free = [t for e, t in c[::3] if ok(e, False)]
print(f"{top} d>={dmin}: {len(c)} candidatos, muestreo {len(c[::3])}: alcanzables sin colisión-check {len(free)}")
for t in free[:5]:
    e = t @ np.linalg.inv(X0)
    print("  cámara", np.round(t[:3, 3], 3), " con colisiones:", ok(e, True))

from moveit_msgs.srv import GetStateValidity
sv = n.create_client(GetStateValidity, "/check_state_validity"); sv.wait_for_service()
def contacts(T):
    r = GetPositionIK.Request(); q = r.ik_request; q.group_name = "lite6"; q.ik_link_name = "link_eef"
    q.avoid_collisions = False; q.timeout.nanosec = 300_000_000; q.pose_stamped.header.frame_id = "link_base"
    fill_ros_pose(q.pose_stamped.pose, T)
    q.robot_state.joint_state.name = [f"joint{i}" for i in range(1, 7)]; q.robot_state.joint_state.position = [0, 0.2, 1.0, 0, 0.8, 0]
    f = ik.call_async(r); rclpy.spin_until_future_complete(n, f, timeout_sec=5); res = f.result()
    if res.error_code.val != 1: return None
    v = GetStateValidity.Request(); v.robot_state = res.solution; v.group_name = "lite6"
    g = sv.call_async(v); rclpy.spin_until_future_complete(n, g, timeout_sec=5)
    return sorted({(c.contact_body_1, c.contact_body_2) for c in g.result().contacts}), np.round(res.solution.joint_state.position[:6], 2)
for t in free[:6]:
    print("  contactos", contacts(t @ np.linalg.inv(X0)))
