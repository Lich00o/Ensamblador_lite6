"""Utilidades de transformaciones homogéneas 4x4 (numpy + scipy)."""

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def T_from_xyz_quat(xyz, quat_xyzw):
    return make_T(Rotation.from_quat(quat_xyzw).as_matrix(), xyz)


def T_from_xyz_rpy(xyz, rpy):
    return make_T(Rotation.from_euler('xyz', rpy).as_matrix(), xyz)


def T_to_xyz_quat(T):
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


def T_from_rvec_tvec(rvec, tvec):
    return make_T(Rotation.from_rotvec(np.asarray(rvec).ravel()).as_matrix(), tvec)


def rotation_angle(R):
    """Ángulo (rad) de la rotación R."""
    return float(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))


def mean_T(Ts):
    """Promedio de transformaciones: media de traslaciones y media cuaterniónica."""
    Ts = list(Ts)
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    R = Rotation.from_matrix(np.array([T[:3, :3] for T in Ts])).mean().as_matrix()
    return make_T(R, t)


def T_from_ros_transform(tf):
    """geometry_msgs/Transform -> 4x4."""
    return T_from_xyz_quat(
        [tf.translation.x, tf.translation.y, tf.translation.z],
        [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])


def T_from_ros_pose(pose):
    """geometry_msgs/Pose -> 4x4."""
    return T_from_xyz_quat(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])


def fill_ros_pose(pose, T):
    xyz, q = T_to_xyz_quat(T)
    pose.position.x, pose.position.y, pose.position.z = map(float, xyz)
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = map(float, q)
    return pose


def fill_ros_transform(tf, T):
    xyz, q = T_to_xyz_quat(T)
    tf.translation.x, tf.translation.y, tf.translation.z = map(float, xyz)
    tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = map(float, q)
    return tf
