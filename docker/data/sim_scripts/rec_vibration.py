#!/usr/bin/env python3
"""Graba 20 s del seguimiento para diagnosticar vibración: joint_states, comandos, objetivo, pose TCP deseada."""
import pickle, sys, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import PointStamped, PoseStamped

D = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
rclpy.init()
n = Node('rec_vibration')
rec = {'js': [], 'cmd': [], 'tgt': [], 'eef': []}
now = lambda: time.monotonic()
n.create_subscription(JointState, '/ufactory/joint_states', lambda m: rec['js'].append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, list(m.name), list(m.position), list(m.velocity))), 300)
n.create_subscription(JointTrajectory, '/lite6_traj_controller/joint_trajectory',
                      lambda m: rec['cmd'].append((now(), [(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9, list(p.positions), list(p.velocities)) for p in m.points])), 300)
n.create_subscription(PointStamped, '/person_tracker/target_point', lambda m: rec['tgt'].append((now(), m.point.x, m.point.y, m.point.z)), 300)
n.create_subscription(PoseStamped, '/person_tracker/target_eef_pose',
                      lambda m: rec['eef'].append((now(), m.pose.position.x, m.pose.position.y, m.pose.position.z,
                                                   m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w)), 300)
t0 = now()
while now() - t0 < D:
    rclpy.spin_once(n, timeout_sec=0.05)
pickle.dump(rec, open('/home/ubuntu/.ros/lite6_handeye/diag/vib.pkl', 'wb'))
print({k: len(v) for k, v in rec.items()})
