"""Altura máxima de link_eef (mirando abajo) alcanzable sobre puntos de la mesa, con IK + colisiones."""
import sys, numpy as np, rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
sys.path.insert(0, "/ws/src/lite6_handeye")
from lite6_handeye.transforms import T_from_xyz_rpy, fill_ros_pose

CAM_Z = 0.042  # link_eef -> lente (CAD)
rclpy.init(); n = Node("reach_height")
ik = n.create_client(GetPositionIK, "/compute_ik"); ik.wait_for_service()

def ok(xyz):
    r = GetPositionIK.Request(); q = r.ik_request; q.group_name = "lite6"; q.ik_link_name = "link_eef"
    q.avoid_collisions = True; q.timeout.nanosec = 300_000_000; q.pose_stamped.header.frame_id = "link_base"
    fill_ros_pose(q.pose_stamped.pose, T_from_xyz_rpy(xyz, [np.pi, 0, 0]))
    q.robot_state.joint_state.name = [f"joint{i}" for i in range(1, 7)]
    q.robot_state.joint_state.position = [0, 0.2, 1.0, 0, 0.8, 0]
    f = ik.call_async(r); rclpy.spin_until_future_complete(n, f, timeout_sec=5)
    return f.result() is not None and f.result().error_code.val == 1

for x, y in [(0.15, 0.0), (0.20, 0.0), (0.25, 0.0), (0.30, 0.0), (0.24, 0.20), (0.33, -0.20)]:
    best = None
    for zc in np.arange(0.20, 0.90, 0.02):  # altura del lente sobre la mesa (cámara en +y mundo)
        if ok([x, y - 0.070, zc + CAM_Z]): best = zc
    print(f"mesa ({x:.2f},{y:+.2f}): lente a máx {best if best is None else round(best, 2)} m sobre la mesa")
