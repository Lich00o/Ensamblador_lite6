#!/usr/bin/env python3
"""Graba todo el lazo del person_tracker para diagnóstico de fluidez (uso: rec_tracking.py <s> <nombre>)."""
import json, pickle, sys, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String
D = float(sys.argv[1]); name = sys.argv[2]
rclpy.init(); n = Node('rec_tracking')
st = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9  # noqa: E731
now = lambda: n.get_clock().now().nanoseconds * 1e-9  # noqa: E731
R = {'js': [], 'cmd': [], 'tgt': [], 'eef': [], 'tracks': [], 'rs': []}
J = [f'joint{i}' for i in range(1, 7)]
def js(m):
    d = dict(zip(m.name, m.position)); v = dict(zip(m.name, m.velocity)) if m.velocity else {}
    R['js'].append((st(m.header), [d[j] for j in J], [v.get(j, 0.0) for j in J]))
def cmd(m):
    R['cmd'].append((now(), [(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9, list(p.positions), list(p.velocities)) for p in m.points]))
n.create_subscription(JointState, '/ufactory/joint_states', js, 500)
n.create_subscription(JointTrajectory, '/lite6_traj_controller/joint_trajectory', cmd, 200)
n.create_subscription(PointStamped, '/person_tracker/target_point', lambda m: R['tgt'].append((st(m.header), now(), m.point.x, m.point.y, m.point.z)), 100)
n.create_subscription(PoseStamped, '/person_tracker/target_eef_pose', lambda m: R['eef'].append((st(m.header), now(), m.pose.position.x, m.pose.position.y, m.pose.position.z, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w)), 100)
n.create_subscription(String, '/person_tracker/tracks', lambda m: R['tracks'].append((now(), m.data)), 100)
try:
    from xarm_msgs.msg import RobotMsg
    n.create_subscription(RobotMsg, '/ufactory/robot_states', lambda m: R['rs'].append((now(), m.state, m.mode, m.err)), 50)
except ImportError:
    pass
t0 = time.monotonic()
while time.monotonic() - t0 < D:
    rclpy.spin_once(n, timeout_sec=0.01)
pickle.dump(R, open(f'/home/ubuntu/.ros/lite6_handeye/diag/{name}.pkl', 'wb'))
print({k: len(v) for k, v in R.items()})
