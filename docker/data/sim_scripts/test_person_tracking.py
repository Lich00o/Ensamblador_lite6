"""Prueba y evaluación en Gazebo del seguimiento 6-DOF de una persona en movimiento.

Mueve `person_target` por varias posiciones 3D (izquierda, derecha, arriba/cerca, centro)
usando `/world/handeye/set_pose` y mide en cada punto:
  - La detección en `/person_tracker/debug_image` y `/person_tracker/target_point`
  - La traslación 3D de la cámara (`cam_pos` en `link_base`)
  - El error angular (°) entre el eje óptico (+z) y la persona real, y el error en píxeles
Guarda una captura anotada en `~/.ros/lite6_handeye/sim_person_track.png`.
"""

import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

sys.path.insert(0, '/ws/src/lite6_handeye')
from lite6_handeye import calib_io, person_tracker

WAYPOINTS = [
    ('centro_inicial', (1.35, 0.00, 0.00)),
    ('izquierda',      (1.35, 0.40, 0.00)),
    ('derecha',        (1.35, -0.40, 0.00)),
    ('diagonal_alta',  (1.25, 0.25, 0.12)),
    ('centro_final',   (1.35, 0.00, 0.00)),
]


def set_person_pose(x: float, y: float, z: float):
    req = f'name: "person_target", position: {{x: {x}, y: {y}, z: {z}}}'
    subprocess.run(
        ['gz', 'service', '-s', '/world/handeye/set_pose',
         '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '2000', '--req', req],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    rclpy.init()
    node = Node('eval_person_tracking')
    state = {}

    node.create_subscription(
        PointStamped, '/person_tracker/target_point',
        lambda m: state.update(pt=np.array([m.point.x, m.point.y, m.point.z]), t_pt=time.time()), 10)
    node.create_subscription(
        PoseStamped, '/person_tracker/target_eef_pose',
        lambda m: state.update(eef=m), 10)
    node.create_subscription(
        Image, '/person_tracker/debug_image',
        lambda m: state.update(dbg=m), 2)

    def _on_js(m: JointState):
        d = dict(zip(m.name, m.position))
        if len(d) >= 6:
            state['q'] = np.array([d[f'joint{i}'] for i in range(1, 7)], dtype=float)

    node.create_subscription(JointState, '/joint_states', _on_js, 20)

    cal = calib_io.load_result('/home/ubuntu/.ros/lite6_handeye/lite6_zed_calib.yaml')
    X = calib_io.tf_dict_to_T(cal['eef_to_optical'])

    # Espera primera detección tras alcanzar WATCH_JOINTS
    t0 = time.time()
    while time.time() - t0 < 25.0 and 'pt' not in state:
        rclpy.spin_once(node, timeout_sec=0.1)
    if 'pt' not in state:
        print('ERROR: no se recibió /person_tracker/target_point')
        sys.exit(1)

    errors_deg = []
    cam_positions = []
    saved_snap = False

    for label, (px, py, pz) in WAYPOINTS:
        set_person_pose(px, py, pz)
        # Espera 3.5 s mientras el seguidor 6-DOF reorienta y traslada el brazo
        t_end = time.time() + 3.5
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)

        q = state['q']
        T_cam = person_tracker.lite6_fk(q) @ X
        cam_pos = T_cam[:3, 3]
        cam_z_axis = T_cam[:3, 2]
        # El punto de apuntamiento en person_target (cuello/barbilla) está a z_modelo + 0.37 m
        true_aim = np.array([px, py, pz + 0.37])
        ray = true_aim - cam_pos
        ray /= np.linalg.norm(ray)
        ang_err_deg = float(np.rad2deg(np.arccos(np.clip(np.dot(cam_z_axis, ray), -1.0, 1.0))))
        errors_deg.append(ang_err_deg)
        cam_positions.append(cam_pos.copy())

        pt_est = state.get('pt', np.zeros(3))
        print(f'[{label:14s}] persona=({px:.2f},{py:+.2f},{pz + 0.37:.2f}) '
              f'est=({pt_est[0]:.2f},{pt_est[1]:+.2f},{pt_est[2]:.2f}) '
              f'cam_pos=({cam_pos[0]:.3f},{cam_pos[1]:+.3f},{cam_pos[2]:.3f}) '
              f'err_apuntamiento={ang_err_deg:.2f}°')

        if label == 'izquierda' and 'dbg' in state and not saved_snap:
            m = state['dbg']
            arr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
            cv2.imwrite('/home/ubuntu/.ros/lite6_handeye/sim_person_track.png', arr)
            saved_snap = True

    cam_arr = np.array(cam_positions)
    span_y_mm = float(np.ptp(cam_arr[:, 1]) * 1000.0)
    span_z_mm = float(np.ptp(cam_arr[:, 2]) * 1000.0)
    mean_err = float(np.mean(errors_deg))
    max_err = float(np.max(errors_deg))
    print(f'=== RESUMEN SEGUIMIENTO 6-DOF ===')
    print(f'Traslación 3D de cámara: Δy = {span_y_mm:.1f} mm, Δz = {span_z_mm:.1f} mm')
    print(f'Error angular de apuntamiento: media = {mean_err:.2f}°, máx = {max_err:.2f}°')
    assert mean_err < 5.0 and span_y_mm > 50.0, 'El seguimiento 6-DOF no alcanzó la precisión esperada'
    print('SEGUIMIENTO 6-DOF ACEPTADO')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
