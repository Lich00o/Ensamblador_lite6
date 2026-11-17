"""Nodo ROS 2 de detección de personas y seguimiento 6-DOF (Eye-in-Hand).

Flujo en tiempo real:
  1. Al arrancar (con `enable_motion:=true`), lleva el Lite 6 a la pose de vigilancia
     horizontal (`WATCH_JOINTS`), donde la cámara mira hacia +x de `link_base`.
  2. Obtiene la calibración Hand-Eye `X = T_eef_optical` desde TF (`link_eef -> optical_frame`),
     o del archivo YAML (`calibration_file`), o de `x_prior_xyz_rpy`.
  3. En cada imagen RGB detecta a la persona (`PersonDetector`), reconstruye su posición 3D
     en `link_base` usando los intrínsecos `K` y la transformación calibrada `X`, y la filtra
     con un filtro de Kalman de velocidad constante (`kf_*`), que descarta saltos.
  4. Calcula la pose 6-DOF deseada (`compute_tracking_pose_6dof`), que traslada la cámara en
     3D (acimut, radio y altura) y orienta su eje óptico directamente hacia la persona.
  5. Resuelve la cinemática inversa 6-DOF (`lite6_ik_6dof`) sembrada en la postura actual. La
     solución es la meta de `JointSmoother` (Ruckig: velocidad, aceleración y jerk limitados),
     que a `stream_rate_hz` publica una referencia continua en
     `/lite6_traj_controller/joint_trajectory` (sin frenar entre metas).
  6. Publica `~/debug_image`, `~/target_point` (`PointStamped`) y `~/target_eef_pose` (`PoseStamped`).

Control manual: `~/set_tracking` (std_srvs/SetBool) activa o pausa el seguimiento (la detección y la
imagen de debug siguen; al pausar, el brazo frena en 0.2 s y queda libre para moverlo a mano);
`~/go_watch` (std_srvs/Trigger) vuelve a la pose de vigilancia. Estado en `~/tracking_enabled`.

IDs por persona (`id_tracking`): ByteTrack asigna un ID estable a cada persona y solo se sigue el
objetivo elegido. Selección: clic sobre la persona en `~/debug_image` (rqt_image_view publica
`~/debug_image_mouse_left`), o `~/select_target` (std_msgs/Int32: ID >= 0, -1 = la más cercana,
-2 = soltar). Lista de tracks y objetivo en `~/tracks` (std_msgs/String, JSON).
"""

import json
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from builtin_interfaces.msg import Duration as MsgDuration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Int32, String
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener
try:
    from xarm_msgs.msg import RobotMsg
except ImportError:  # sin xarm_ros2 (p. ej. pruebas)
    RobotMsg = None
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import byte_tracker, calib_io, person_tracker
from .image_conv import bgr_to_msg, image_to_numpy, to_bgr
from .motion_smoother import JointSmoother
from .transforms import T_from_ros_transform, T_from_xyz_rpy, fill_ros_pose

JOINT_NAMES = [f'joint{i}' for i in range(1, 7)]


class PersonTrackerNode(Node):
    def __init__(self):
        super().__init__('person_tracker')
        d = self.declare_parameter
        d('image_topic', '/sim_camera/image')
        d('camera_info_topic', '/sim_camera/camera_info')
        d('joint_states_topic', '/joint_states')  # robot real: /ufactory/joint_states (150 Hz; /joint_states va a 10 Hz)
        d('image_latency_s', 0.03)         # captura -> stamp de la imagen; la pose de la cámara se toma en ese instante
        d('robot_states_topic', '/ufactory/robot_states')  # '' = no verificar (Gazebo)
        d('trajectory_action', '/lite6_traj_controller/follow_joint_trajectory')
        d('robot_base_frame', 'link_base')
        d('robot_effector_frame', 'link_eef')
        d('optical_frame', 'sim_camera_optical_frame')
        d('calibration_file', '~/.ros/lite6_handeye/lite6_zed_calib.yaml')
        d('x_prior_xyz_rpy', [0.0125, -0.070, 0.042, 0.0, 0.0, 3.14159265])
        d('onnx_model', '')                # YOLO ONNX (p. ej. models/yolo26n.onnx); '' = heurísticas
        d('yolo_min_score', 0.45)          # confianza mínima de la clase 'person' con YOLO
        d('ref_torso_height_m', 0.52)
        d('default_person_distance', 1.15)
        d('use_fixed_distance', False)
        d('kf_accel_std', 0.5)             # Kalman: aceleración blanca de la persona (m/s^2)
        d('kf_meas_lat_std', 0.03)         # Kalman: ruido de medición perpendicular al rayo (m)
        d('kf_meas_depth_std', 0.15)       # Kalman: ruido de medición a lo largo del rayo (m)
        d('kf_gate_sigma', 3.5)            # Kalman: descarta mediciones a > N sigma
        d('kf_max_rejects', 5)             # Kalman: descartes seguidos antes de reiniciar
        d('cam_radius_nom', 0.30)          # radio xy nominal de la cámara (m)
        d('cam_radius_gain', 0.05)         # traslación radial según distancia de la persona
        d('cam_z_nom', 0.38)               # altura nominal de la cámara en WATCH_JOINTS (m)
        d('cam_z_gain', 0.28)              # traslación vertical según altura de la persona
        d('max_azimuth_deg', 70.0)
        d('cam_radius_range_m', [0.24, 0.34])  # radio xy permitido de la cámara (m)
        d('cam_z_range_m', [0.28, 0.46])       # altura permitida de la cámara (m)
        d('enable_motion', True)
        d('tracking_enabled', True)        # estado inicial del seguimiento (cambia con ~/set_tracking)
        d('id_tracking', True)             # ByteTrack: ID por persona; se sigue solo el objetivo elegido
        d('auto_select_target', False)     # sin objetivo elegido: tomar la persona más cercana
        d('track_high_thresh', 0.5)        # ByteTrack: detección 'alta' (1.ª asociación)
        d('track_low_thresh', 0.15)        # ByteTrack: detección 'baja' (2.ª asociación, oclusiones)
        d('track_buffer_s', 3.0)           # ByteTrack: tiempo que un ID perdido se conserva
        d('camera_fps', 30.0)              # cuadros/s de la cámara (convierte track_buffer_s a cuadros)
        d('reid_enabled', True)            # galería de IDs: quien sale de cuadro vuelve con su ID (ropa)
        d('reid_buffer_s', 120.0)          # cuánto recuerda la galería un ID expirado
        d('reid_model', '')                # ReID ONNX (OSNet-AIN); '' = histograma de color (poco fiable)
        d('reid_thresh', 0.6)              # similitud mínima (coseno con reid_model; Bhattacharyya sin él)
        d('tracker_cmc', True)             # compensar el giro de la cámara (cinemática) en el tracker
        d('move_to_watch_on_start', True)
        d('watch_joints_deg', [0.0])       # pose de vigilancia (6 valores, grados); [0.0] = WATCH_JOINTS
        d('control_period', 0.08)          # s entre comandos de seguimiento (~12.5 Hz)
        d('max_joint_vel_rad_s', 1.80)     # velocidad articular máxima en seguimiento
        d('max_joint_acc_rad_s2', 1.5)     # aceleración articular máxima (suavizado)
        d('max_joint_jerk_rad_s3', 8.0)    # jerk articular máximo (Ruckig)
        d('smooth_motion', True)           # False = trayectorias de un punto (comportamiento previo)
        d('stream_rate_hz', 50.0)          # frecuencia de la referencia suavizada
        d('stream_lookahead_s', 0.10)      # horizonte de cada punto enviado al controlador
        d('joint_trajectory_topic', '/lite6_traj_controller/joint_trajectory')
        d('min_move_time', 0.14)           # duración mínima de cada segmento (s)
        d('joint_deadband_rad', 0.005)     # umbral mínimo para enviar comando (~0.3°)
        d('deadband_enter_deg', 5.0)       # quieto mientras la cabeza esté a < N° del eje óptico
        d('deadband_exit_deg', 2.0)        # al salir, recentra hasta quedar a < N° (histéresis)
        d('max_wrist_jump_deg', 25.0)      # descarta IK que gire J4 o J6 más (cambio de rama de muñeca)
        d('lost_timeout', 5.0)             # s sin detección antes de volver a WATCH_JOINTS
        d('publish_debug_image', True)

        self.P = lambda n: self.get_parameter(n).value  # noqa: E731
        self.base_frame = self.P('robot_base_frame')
        self.eef_frame = self.P('robot_effector_frame')
        self.optical_frame = self.P('optical_frame')

        self.detector = person_tracker.PersonDetector(
            onnx_model=self.P('onnx_model'),
            ref_torso_height_m=self.P('ref_torso_height_m'),
            min_score=float(self.P('yolo_min_score')) if self.P('onnx_model') else 0.25,
        )
        if self.detector.net is not None:
            self.get_logger().info(f'Detector: {self.P("onnx_model")} ({self.detector.net_provider})')
        self.X_eef_opt = self._load_initial_calibration()
        self.tf_calibrated_logged = False

        cb = ReentrantCallbackGroup()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.traj_client = ActionClient(
            self, FollowJointTrajectory, self.P('trajectory_action'), callback_group=cb)

        self.pub_point = self.create_publisher(PointStamped, '~/target_point', 10)
        self.pub_pose = self.create_publisher(PoseStamped, '~/target_eef_pose', 10)
        self.pub_dbg = self.create_publisher(Image, '~/debug_image', 2)
        self.pub_traj = self.create_publisher(JointTrajectory, self.P('joint_trajectory_topic'), 10)

        self.lock = threading.Lock()
        self.K = None
        self.joints = None
        self.prev_uv = None
        self.target_base = None
        self.last_det_time = None
        self.centering = False
        self.kf = person_tracker.TargetKalman(
            accel_std=self.P('kf_accel_std'), meas_lat_std=self.P('kf_meas_lat_std'),
            meas_depth_std=self.P('kf_meas_depth_std'), gate_sigma=self.P('kf_gate_sigma'),
            max_rejects=self.P('kf_max_rejects'))
        self.last_cmd_time = 0.0
        self.last_cmd_q = None
        self.motion_lock = threading.Lock()
        self.smoother = JointSmoother(
            6, 1.0 / float(self.P('stream_rate_hz')), float(self.P('max_joint_vel_rad_s')),
            float(self.P('max_joint_acc_rad_s2')), float(self.P('max_joint_jerk_rad_s3')))
        self.stream_idle = True
        self.js_hist = deque(maxlen=400)      # (t, q): ~2.7 s a 150 Hz
        self.tracking = bool(self.P('tracking_enabled'))
        self.add_on_set_parameters_callback(self._on_params)
        self.byte = byte_tracker.ByteTracker(
            high_thresh=float(self.P('track_high_thresh')), low_thresh=float(self.P('track_low_thresh')),
            new_track_thresh=float(self.P('track_high_thresh')) + 0.1,
            track_buffer=int(float(self.P('track_buffer_s')) * float(self.P('camera_fps'))),
            reid=bool(self.P('reid_enabled')), reid_thresh=float(self.P('reid_thresh')),
            reid_probation=int(float(self.P('camera_fps'))),          # ~1 s para decidir si es conocido
            similarity=byte_tracker.cosine_similarity if self.P('reid_model')
            else byte_tracker.appearance_similarity,
            reid_buffer=int(float(self.P('reid_buffer_s')) * float(self.P('camera_fps'))))
        self.reid_embedder = None
        if self.P('reid_enabled') and self.P('reid_model'):
            self.reid_embedder = byte_tracker.ReIDEmbedder(self.P('reid_model'))
            self.get_logger().info(f'ReID: {self.P("reid_model")} ({self.reid_embedder.provider})')
        self.target_id = None
        self.tracks = []
        self.prev_R_cam = None
        self.robot_ok = True            # sin topic de estado se asume listo
        self.robot_status = ''
        self.watch_reached = not bool(self.P('move_to_watch_on_start'))
        self.initializing = False
        wj = [float(v) for v in self.P('watch_joints_deg')]
        self.q_watch = (np.deg2rad(wj) if len(wj) == 6
                        else np.array(person_tracker.WATCH_JOINTS, dtype=float))
        T_cw = person_tracker.lite6_fk(self.q_watch) @ self.X_eef_opt
        self.az_watch = float(np.arctan2(T_cw[1, 3], T_cw[0, 3]))   # acimut de la cámara en vigilancia

        self.create_subscription(
            CameraInfo, self.P('camera_info_topic'), self._on_info, qos_profile_sensor_data,
            callback_group=cb)
        self.create_subscription(
            JointState, self.P('joint_states_topic'), self._on_js, 50, callback_group=cb)
        self.create_subscription(
            Image, self.P('image_topic'), self._on_image, qos_profile_sensor_data,
            callback_group=cb)
        if self.P('robot_states_topic') and RobotMsg is not None:
            self.robot_ok = False
            self.create_subscription(RobotMsg, self.P('robot_states_topic'), self._on_robot_state, 10,
                                     callback_group=cb)
        self.create_timer(0.25, self._on_watchdog, callback_group=cb)
        self.pub_tracking = self.create_publisher(
            Bool, '~/tracking_enabled', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub_tracking.publish(Bool(data=self.tracking))
        self.create_service(SetBool, '~/set_tracking', self._on_set_tracking, callback_group=cb)
        self.create_service(Trigger, '~/go_watch', self._on_go_watch, callback_group=cb)
        self.pub_tracks = self.create_publisher(String, '~/tracks', 5)
        self.create_subscription(Int32, '~/select_target', self._on_select_id, 10, callback_group=cb)
        self.create_subscription(Point, '~/debug_image_mouse_left', self._on_click, 10, callback_group=cb)
        if self.P('smooth_motion'):
            self.create_timer(1.0 / float(self.P('stream_rate_hz')), self._on_stream,
                              callback_group=MutuallyExclusiveCallbackGroup())

        self.get_logger().info(
            f'person_tracker listo: image={self.P("image_topic")}, optical={self.optical_frame}, '
            f'movimiento={self.smoother.backend if self.P("smooth_motion") else "puntos"}')

    def _load_initial_calibration(self) -> np.ndarray:
        cal_path = calib_io.expand(self.P('calibration_file'))
        if cal_path and os.path.exists(cal_path):
            try:
                res = calib_io.load_result(cal_path)
                X = calib_io.tf_dict_to_T(res['eef_to_optical'])
                self.get_logger().info(f'Calibración Hand-Eye cargada de {cal_path}')
                return X
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f'No se pudo leer {cal_path}: {e}')
        p = self.P('x_prior_xyz_rpy')
        self.get_logger().info('Usando x_prior_xyz_rpy para T_eef_optical')
        return T_from_xyz_rpy(p[:3], p[3:])

    def _update_tf_calibration(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.eef_frame, self.optical_frame, Time())
            self.X_eef_opt = T_from_ros_transform(tf.transform)
            if not self.tf_calibrated_logged:
                self.tf_calibrated_logged = True
                self.get_logger().info(
                    f'Usando TF calibrada en vivo: {self.eef_frame} -> {self.optical_frame}')
        except Exception:  # noqa: BLE001
            pass

    def _current_T_base_eef(self, q_now: np.ndarray) -> np.ndarray:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.eef_frame, Time())
            return T_from_ros_transform(tf.transform)
        except Exception:  # noqa: BLE001
            return person_tracker.lite6_fk(q_now)

    def _on_info(self, msg: CameraInfo):
        with self.lock:
            self.K = np.asarray(msg.k, dtype=float).reshape(3, 3)

    def _on_js(self, msg: JointState):
        mapping = dict(zip(msg.name, msg.position))
        if all(n in mapping for n in JOINT_NAMES):
            q = np.array([float(mapping[n]) for n in JOINT_NAMES], dtype=float)
            st = msg.header.stamp
            t = st.sec + st.nanosec * 1e-9 if (st.sec or st.nanosec) else self.get_clock().now().nanoseconds * 1e-9
            with self.lock:
                self.joints = q
                self.js_hist.append((t, q))

    def _q_at(self, t: float):
        """Articulaciones interpoladas en el instante t (s, reloj ROS); None sin historial que lo cubra."""
        with self.lock:
            h = list(self.js_hist)
        if len(h) < 2 or t < h[0][0]:
            return None
        if t >= h[-1][0]:
            return h[-1][1] if t - h[-1][0] < 0.05 else None
        ts = np.array([x[0] for x in h])
        k = int(np.searchsorted(ts, t))
        (t0, q0), (t1, q1) = h[k - 1], h[k]
        a = (t - t0) / max(t1 - t0, 1e-9)
        return q0 + a * (q1 - q0)

    def _send_trajectory(self, q_target: np.ndarray, duration_sec: float):
        if not self.P('enable_motion'):
            return
        if not self.traj_client.server_is_ready():
            return
        sec = int(duration_sec)
        nanosec = int(round((duration_sec - sec) * 1e9))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q_target]
        pt.time_from_start = MsgDuration(sec=sec, nanosec=nanosec)
        goal.trajectory.points = [pt]
        self.traj_client.send_goal_async(goal)
        self.last_cmd_time = time.monotonic()
        self.last_cmd_q = q_target.copy()

    def _on_set_tracking(self, req, res):
        self.tracking = bool(req.data)
        self.centering = False
        self.pub_tracking.publish(Bool(data=self.tracking))
        # al pausar se deja de transmitir: el último mensaje trae un punto de frenado (0.2 s, v = 0),
        # el brazo queda libre para el jog manual y al reactivar se parte de la postura medida
        with self.motion_lock:
            self.smoother.stop()
        self.last_cmd_q = None
        res.success, res.message = True, 'seguimiento ' + ('ACTIVADO' if self.tracking else 'EN PAUSA')
        self.get_logger().info(res.message)
        return res

    def _on_params(self, params):
        """Aplica en caliente los parámetros que solo se leían al crear el suavizador y el Kalman
        (para afinar con `ros2 param set` sin reiniciar el nodo)."""
        lim = {'max_joint_vel_rad_s': 'max_vel', 'max_joint_acc_rad_s2': 'max_acc',
               'max_joint_jerk_rad_s3': 'max_jerk'}
        kf = {'kf_accel_std': 'accel_std', 'kf_meas_lat_std': 'meas_lat_std',
              'kf_meas_depth_std': 'meas_depth_std', 'kf_gate_sigma': 'gate_sigma'}
        for p in params:
            if p.name in lim:
                with self.motion_lock:
                    self.smoother.set_limits(**{lim[p.name]: p.value})
            elif p.name in kf:
                setattr(self.kf, kf[p.name], float(p.value))
        return SetParametersResult(successful=True)

    def _set_target(self, tid, why: str):
        if tid == self.target_id:
            return
        self.target_id = tid
        self.kf.reset()                      # objetivo nuevo: no mezclar con la posición del anterior
        self.target_base = None
        self.prev_uv = None
        self.centering = False
        self.last_det_time = None
        self.get_logger().info(f'Objetivo: {"ninguno" if tid is None else f"ID {tid}"} ({why})')

    def _on_select_id(self, msg):
        with self.lock:
            tracks = list(self.tracks)
        if msg.data == -2:
            self._set_target(None, 'soltado')
        elif msg.data == -1:
            t = byte_tracker.select_nearest(tracks)
            self._set_target(None if t is None else t.id, 'la más cercana')
        elif any(t.id == msg.data for t in tracks):
            self._set_target(int(msg.data), 'elegido por ID')
        else:
            self.get_logger().warn(f'ID {msg.data} no existe')

    def _on_click(self, msg):
        with self.lock:
            tracks = list(self.tracks)
        t = byte_tracker.select_at(tracks, msg.x, msg.y)
        if t is None:
            self._set_target(None, 'clic fuera de personas')
        else:
            self._set_target(t.id, 'clic')

    def _on_go_watch(self, req, res):
        if not self.robot_ok:
            res.success, res.message = False, f'robot no listo ({self.robot_status})'
            return res
        with self.motion_lock:
            self.smoother.stop()
        self.last_cmd_q = None
        self.centering = False
        self.watch_reached = False          # el watchdog la envía (y confirma la llegada)
        res.success, res.message = True, 'yendo a la pose de vigilancia'
        self.get_logger().info(res.message)
        return res

    def _on_robot_state(self, msg):
        # ros2_control (lite6_traj_controller) solo corre en modo servo (1) sin error
        ok = msg.mode == 1 and msg.err == 0 and msg.state in (0, 1, 2)
        self.robot_status = f'mode {msg.mode}, state {msg.state}, err {msg.err}'
        if self.robot_ok and not ok:
            self.get_logger().warn(f'Robot no listo ({self.robot_status}): movimiento en pausa. '
                                   'Volver a mode 1 / state 0 (set_mode, set_state).')
            with self.motion_lock:
                self.smoother.stop()        # al volver, reinicia desde la postura medida
            self.last_cmd_q = None
            self.watch_reached = not bool(self.P('move_to_watch_on_start'))
        elif ok and not self.robot_ok:
            self.get_logger().info('Robot listo (mode 1): movimiento reanudado.')
        self.robot_ok = ok

    def _on_stream(self):
        """Publica la referencia suavizada (posición + velocidad) a `stream_rate_hz`."""
        if not self.P('enable_motion'):
            return
        if not self.robot_ok:
            return
        la = float(self.P('stream_lookahead_s'))
        with self.motion_lock:
            if not self.smoother.active:
                return
            if self.smoother.settled():
                if self.stream_idle:
                    return              # meta alcanzada y ya enviada: el controlador la mantiene
                self.stream_idle = True
            else:
                self.stream_idle = False
            _, _, q_la, v_la = self.smoother.step(lookahead=la)
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        # el controlador exige velocidad 0 en el último punto (allow_nonzero_velocity_at_trajectory_end
        # = false): tras el punto suavizado se agrega uno de frenado lineal. Como llega un mensaje
        # nuevo cada 1/stream_rate_hz, solo se ejecuta el tramo inicial (continuo en velocidad).
        t_stop = 0.2
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q_la]
        pt.velocities = [float(v) for v in v_la]
        pt.time_from_start = MsgDuration(sec=0, nanosec=int(la * 1e9))
        end = JointTrajectoryPoint()
        end.positions = [float(q + 0.5 * v * t_stop) for q, v in zip(q_la, v_la)]
        end.velocities = [0.0] * len(JOINT_NAMES)
        end.time_from_start = MsgDuration(sec=0, nanosec=int((la + t_stop) * 1e9))
        msg.points = [pt, end]
        self.pub_traj.publish(msg)

    def _watch_duration(self, q_now) -> float:
        """Duración del viaje a la pose de vigilancia: velocidad media <= 2/3 de max_joint_vel
        (el spline de un punto tiene un pico ~1.5x la media)."""
        dist = float(np.max(np.abs(self.q_watch - q_now)))
        return max(2.5, 1.5 * dist / float(self.P('max_joint_vel_rad_s')))

    def _on_watchdog(self):
        if not self.P('enable_motion'):
            return
        with self.lock:
            q_now = None if self.joints is None else self.joints.copy()
        if q_now is None:
            return
        if not self.robot_ok:
            self.get_logger().warn(f'Esperando robot en mode 1 ({self.robot_status or "sin estado"})',
                                   throttle_duration_sec=5.0)
            return
        if not self.watch_reached and not self.initializing:
            if not self.traj_client.server_is_ready():
                return
            self.initializing = True
            q_watch = self.q_watch
            dur = self._watch_duration(q_now)
            self.get_logger().info(f'Moviendo a pose de vigilancia WATCH_JOINTS ({dur:.1f} s)...')
            self._send_trajectory(q_watch, dur)

            def _mark_ready():
                time.sleep(dur + 0.3)
                with self.lock:
                    q = None if self.joints is None else self.joints.copy()
                err = float('inf') if q is None else float(np.max(np.abs(q - q_watch)))
                self.initializing = False
                if err > np.deg2rad(3.0):
                    self.get_logger().warn(
                        f'No llegó a la pose de vigilancia (error {np.rad2deg(err):.0f}°); reintentando')
                    return
                self.watch_reached = True
                self.get_logger().info('Pose de vigilancia alcanzada; seguimiento 6-DOF activo.')

            threading.Thread(target=_mark_ready, daemon=True).start()
            return

        if self.tracking and self.watch_reached and self.last_det_time is not None:
            if time.monotonic() - self.last_det_time > float(self.P('lost_timeout')):
                self.last_det_time = None
                self.prev_uv = None
                self.target_base = None
                self.kf.reset()
                self.centering = False
                q_watch = self.q_watch
                self.get_logger().info('Persona fuera de vista; volviendo a WATCH_JOINTS')
                with self.motion_lock:
                    self.smoother.stop()
                self._send_trajectory(q_watch, self._watch_duration(q_now))

    def _on_image(self, msg: Image):
        with self.lock:
            K = None if self.K is None else self.K.copy()
            q_now = None if self.joints is None else self.joints.copy()
        if K is None or q_now is None:
            return
        try:
            raw, enc = image_to_numpy(msg)
            bgr = to_bgr(raw, enc)
        except ValueError:
            return

        self._update_tf_calibration()
        # pose de la cámara en el instante de captura (no la última): con el brazo girando, una pose
        # atrasada desplaza el rayo a la persona y el lazo oscila
        st = msg.header.stamp
        q_img = self._q_at(st.sec + st.nanosec * 1e-9 - float(self.P('image_latency_s')))
        T_base_eef_img = (person_tracker.lite6_fk(q_img) if q_img is not None
                          else self._current_T_base_eef(q_now))
        tracks = None
        if self.P('id_tracking'):
            dets = self.detector.detect_all(bgr, K, float(self.P('track_low_thresh')))
            if self.reid_embedder is not None:
                for dd, f in zip(dets, self.reid_embedder(bgr, [dd.bbox for dd in dets])):
                    dd.feat = f
            elif self.P('reid_enabled'):
                for dd in dets:
                    dd.feat = byte_tracker.appearance_feature(bgr, dd.bbox)
            if self.P('tracker_cmc'):
                # rotación de la cámara desde el cuadro anterior -> homografía de píxeles
                R_cam = (T_base_eef_img @ self.X_eef_opt)[:3, :3]
                if self.prev_R_cam is not None:
                    R_rel = R_cam.T @ self.prev_R_cam             # dirección anterior -> actual
                    if np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec()) > np.radians(0.1):
                        self.byte.apply_homography(K @ R_rel @ np.linalg.inv(K))
                self.prev_R_cam = R_cam
            tracks = self.byte.update(dets)
            with self.lock:
                self.tracks = tracks
            if self.target_id in self.byte.aliases:          # el objetivo recuperó su ID anterior
                self.target_id = self.byte.aliases[self.target_id]
            if self.target_id is not None and not self.byte.known(self.target_id):
                self._set_target(None, 'ID expirado')
            if self.target_id is None and self.P('auto_select_target'):
                t = byte_tracker.select_nearest(tracks)
                if t is not None:
                    self._set_target(t.id, 'auto')
            tgt = self.byte.get(self.target_id)
            det = tgt.det if (tgt is not None and tgt.visible) else None
            self.pub_tracks.publish(String(data=json.dumps({
                'target': self.target_id,
                'tracks': [{'id': t.id, 'visible': t.visible, 'score': round(float(t.det.score), 2),
                            'bbox': [int(v) for v in t.bbox]} for t in tracks],
                'gallery': sorted(self.byte.gallery)})))
        else:
            det = self.detector.detect(bgr, K, self.prev_uv)

        if det is not None and self.watch_reached:
            self.prev_uv = det.track_uv
            self.last_det_time = time.monotonic()
            dist = (float(self.P('default_person_distance'))
                    if self.P('use_fixed_distance') else det.est_distance)
            T_base_eef = T_base_eef_img
            p_meas = person_tracker.ray_to_target_3d(
                det.track_uv[0], det.track_uv[1], K, T_base_eef, self.X_eef_opt, dist)
            cam_origin = (T_base_eef @ self.X_eef_opt)[:3, 3]
            if not self.kf.update(p_meas, time.monotonic(), ray_dir=p_meas - cam_origin):
                self.get_logger().debug('medición descartada por el filtro de Kalman')
            self.target_base = self.kf.position

            T_cam_now = T_base_eef @ self.X_eef_opt
            az_now = float(np.arctan2(T_cam_now[1, 3], T_cam_now[0, 3]))
            T_eef_des, T_cam_des = person_tracker.compute_tracking_pose_6dof(
                self.target_base,
                self.X_eef_opt,
                cam_radius_nom=float(self.P('cam_radius_nom')),
                cam_radius_gain=float(self.P('cam_radius_gain')),
                cam_z_nom=float(self.P('cam_z_nom')),
                cam_z_gain=float(self.P('cam_z_gain')),
                max_azimuth_deg=float(self.P('max_azimuth_deg')),
                cam_radius_range=tuple(float(v) for v in self.P('cam_radius_range_m')),
                cam_z_range=tuple(float(v) for v in self.P('cam_z_range_m')),
                az_ref=az_now,
            )

            pt_msg = PointStamped()
            pt_msg.header.stamp = msg.header.stamp
            pt_msg.header.frame_id = self.base_frame
            pt_msg.point.x, pt_msg.point.y, pt_msg.point.z = map(float, self.target_base)
            self.pub_point.publish(pt_msg)

            pose_msg = PoseStamped()
            pose_msg.header = pt_msg.header
            fill_ros_pose(pose_msg.pose, T_eef_des)
            self.pub_pose.publish(pose_msg)

            # zona muerta con histéresis sobre el ángulo eje óptico -> objetivo
            T_base_cam = T_base_eef @ self.X_eef_opt
            v = self.target_base - T_base_cam[:3, 3]
            ang = float(np.degrees(np.arccos(np.clip(
                T_base_cam[:3, 2] @ v / max(np.linalg.norm(v), 1e-6), -1.0, 1.0))))
            if self.centering and ang < float(self.P('deadband_exit_deg')):
                self.centering = False
            elif not self.centering and ang > float(self.P('deadband_enter_deg')):
                self.centering = True
            if not self.tracking:
                self.centering = False

            now = time.monotonic()
            if self.centering and now - self.last_cmd_time >= float(self.P('control_period')):
                q_seed = self.last_cmd_q if self.last_cmd_q is not None else q_now
                q_sol, ok = person_tracker.lite6_ik_6dof(T_eef_des, q_seed)
                wrist_lim = np.deg2rad(float(self.P('max_wrist_jump_deg')))
                if not ok or np.max(np.abs((q_sol - q_now)[[3, 5]])) > wrist_lim:
                    # 2.º intento: la pose de vigilancia girada en J1 hacia la persona (misma forma
                    # del brazo: J2-J6 no cambian de rama). Resuelve objetivos a la espalda.
                    az_des = float(np.arctan2(T_cam_des[1, 3], T_cam_des[0, 3]))
                    q_rot = self.q_watch.copy()
                    q_rot[0] += az_des - self.az_watch
                    q_rot[0] = q_now[0] + np.arctan2(np.sin(q_rot[0] - q_now[0]), np.cos(q_rot[0] - q_now[0]))
                    q2, ok2 = person_tracker.lite6_ik_6dof(T_eef_des, q_rot)
                    if ok2 and (not ok or np.max(np.abs((q2 - q_now)[[3, 5]]))
                                < np.max(np.abs((q_sol - q_now)[[3, 5]]))):
                        q_sol, ok = q2, ok2
                dq = float(np.max(np.abs(q_sol - q_now))) if ok else 0.0
                d_wrist = float(np.max(np.abs((q_sol - q_now)[[3, 5]]))) if ok else 0.0
                if ok and d_wrist > np.deg2rad(float(self.P('max_wrist_jump_deg'))):
                    # la IK saltó a la otra rama de la muñeca (J4/J6 ~180°): misma vista, giro enorme
                    self.last_cmd_q = None
                    self.get_logger().warn(
                        f'IK descartada: J4/J6 giraría {np.rad2deg(d_wrist):.0f}° > max_wrist_jump_deg',
                        throttle_duration_sec=1.0)
                elif ok:
                    if dq >= float(self.P('joint_deadband_rad')) and self.robot_ok:
                        if self.P('smooth_motion'):
                            with self.motion_lock:
                                if not self.smoother.active:
                                    self.smoother.reset(q_now)
                                self.smoother.set_goal(q_sol)
                            self.last_cmd_q = q_sol.copy()
                            self.last_cmd_time = now
                        else:
                            dur = max(float(self.P('min_move_time')),
                                      dq / float(self.P('max_joint_vel_rad_s')))
                            self._send_trajectory(q_sol, dur)

        if self.P('publish_debug_image') and self.pub_dbg.get_subscription_count() >= 0:
            dbg = person_tracker.draw_debug(bgr, det, self.target_base)
            if tracks is not None:
                byte_tracker_draw(dbg, tracks, self.target_id)
            on = self.tracking and self.robot_ok
            label = ('SEGUIMIENTO ON' if on else 'SEGUIMIENTO PAUSA' if self.tracking is False
                     else f'ROBOT NO LISTO ({self.robot_status})')
            cv2.putText(dbg, label, (18, dbg.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 200, 0) if on else (0, 0, 255), 2, cv2.LINE_AA)
            self.pub_dbg.publish(bgr_to_msg(dbg, msg.header, Image))


def main():
    rclpy.init()
    node = PersonTrackerNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def byte_tracker_draw(img, tracks, target_id):
    """Cajas con ID de todas las personas; el objetivo resaltado (como el HUD de tello_env)."""
    for t in tracks:
        x, y, w, h = (int(v) for v in t.bbox)
        is_t = t.id == target_id
        if not t.visible and not is_t:
            continue
        color = (0, 0, 255) if is_t else (200, 200, 200)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3 if is_t else 1)
        label = (f'>>> OBJETIVO {t.id} <<<' if t.visible else f'OBJETIVO {t.id} PERDIDO') if is_t \
            else f'ID {t.id} {t.det.score:.2f}'
        cv2.putText(img, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if target_id is not None and all(t.id != target_id for t in tracks):
        cv2.putText(img, f'OBJETIVO {target_id} fuera de cuadro (en memoria)', (18, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    if target_id is None:
        cv2.putText(img, 'Sin objetivo: clic sobre una persona', (18, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 200, 255), 2, cv2.LINE_AA)
