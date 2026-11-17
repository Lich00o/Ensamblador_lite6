"""Calibración Eye-in-Hand automática con MoveIt 2.

Secuencia (arranca con: ros2 service call /handeye_auto_calibrator/start std_srvs/srv/Trigger):
  1. Lee pose inicial (articulaciones + TF) y detección del tablero.
  2. Agrega la mesa como CollisionObject (/apply_planning_scene).
  3. Genera N poses mirando al tablero (o reproduce un archivo de poses articulares
     validadas en simulación) y las ejecuta vía la acción /move_action.
  4. En cada pose: espera settle_time, promedia detecciones, verifica quietud.
  5. Vuelve a la pose inicial, resuelve AX = XB y guarda resultado + muestras.

Modo simulación (lite6_moveit_fake, sin cámara): capture:=false y board_pose_base
definido -> solo valida alcanzabilidad/colisiones y guarda <name>_poses.yaml para
reproducir exactamente esas configuraciones en el robot real.
"""

import collections
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (CollisionObject, Constraints, JointConstraint, MoveItErrorCodes,
                             OrientationConstraint, PlanningScene, PositionConstraint)
from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from . import calib_io, pose_sampler
from .charuco import BoardConfig
from .transforms import (T_from_ros_pose, T_from_ros_transform, T_from_xyz_rpy, fill_ros_pose,
                         fill_ros_transform, mean_T)


class AutoCalibrator(Node):
    def __init__(self):
        super().__init__('handeye_auto_calibrator')
        d = self.declare_parameter
        d('name', 'lite6_zed_calib')
        d('output_dir', '~/.ros/lite6_handeye')
        d('robot_base_frame', 'link_base')
        d('robot_effector_frame', 'link_eef')
        d('camera_root_frame', 'zed_camera_link')
        d('board_pose_topic', '/charuco_detector/board_pose')
        d('joint_states_topic', '/joint_states')
        d('move_group', 'lite6')
        d('move_action', '/move_action')
        d('planning_time', 5.0)
        d('planning_attempts', 5)
        d('velocity_scaling', 0.1)
        d('acceleration_scaling', 0.1)
        d('n_poses', 20)
        d('distance_min', 0.22)
        d('distance_max', 0.32)
        d('tilt_min_deg', 20.0)
        d('tilt_max_deg', 45.0)
        d('roll_max_deg', 30.0)
        d('azimuth_range_deg', 100.0)  # +- alrededor de la dirección hacia la base
        d('accept_joint_error_rad', 0.05)  # CONTROL_FAILED cerca del objetivo -> aceptar
        d('wrist_window_deg', 120.0)   # joint4/joint6 a +- de la pose inicial (178 = sin límite extra)
        d('follow_azimuth', True)      # el "arriba" de la cámara gira con el acimut
        d('seed', 0)
        d('min_eef_height', 0.15)      # z mínimo de link_eef sobre la mesa (prefiltro)
        d('settle_time', 1.0)
        d('samples_per_pose', 10)
        d('capture_timeout', 3.0)
        d('max_sample_spread_mm', 2.0)
        d('max_robot_motion_mm', 0.3)
        d('min_samples', 10)
        d('capture', True)
        d('auto_start', False)
        d('poses_file', '')           # si existe: reproduce poses articulares
        d('x_prior_xyz_rpy', [0.0, -0.06, 0.04, 0.0, 0.0, 0.0])  # link_eef -> óptico (CAD)
        d('use_board_pose_base', False)  # simulación sin cámara
        d('board_pose_base', [0.225, 0.105, 0.0, 3.14159265, 0.0, 0.0])  # [x y z r p y] en base
        d('table.add', True)
        d('table.z', 0.0)
        d('table.size_x', 1.2)
        d('table.size_y', 1.2)
        d('table.thickness', 0.05)
        d('table.center_x', 0.3)
        d('table.center_y', 0.0)
        d('method', 'refine')
        d('publish_result', True)
        d('board.squares_x', 5)
        d('board.squares_y', 7)
        d('board.square_length', 0.030)
        d('board.marker_length', 0.022)
        d('board.dictionary', 'DICT_4X4_50')

        self.P = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cfg = BoardConfig(self.P('board.squares_x'), self.P('board.squares_y'),
                               self.P('board.square_length'), self.P('board.marker_length'),
                               self.P('board.dictionary'))
        self.base = self.P('robot_base_frame')
        self.eef = self.P('robot_effector_frame')
        self.out_dir = calib_io.expand(self.P('output_dir'))
        self.name = self.P('name')

        cb = ReentrantCallbackGroup()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.static_tf = StaticTransformBroadcaster(self)
        self.move_client = ActionClient(self, MoveGroup, self.P('move_action'), callback_group=cb)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik', callback_group=cb)
        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene',
                                               callback_group=cb)
        self.dets = collections.deque(maxlen=200)
        self.joints = None
        self.lock = threading.Lock()
        self.create_subscription(PoseStamped, self.P('board_pose_topic'), self._on_det, 50,
                                 callback_group=cb)
        self.create_subscription(JointState, self.P('joint_states_topic'), self._on_js, 50,
                                 callback_group=cb)
        self.create_service(Trigger, '~/start', self._on_start, callback_group=cb)
        self.running = False
        if self.P('auto_start'):
            self.create_timer(2.0, self._auto_start_once, callback_group=cb)
        self.get_logger().info(
            'Listo. Iniciar con: ros2 service call /handeye_auto_calibrator/start std_srvs/srv/Trigger')

    # ---------------------------------------------------------------- callbacks
    def _on_det(self, msg):
        with self.lock:
            # hora de recepción (no header.stamp): robusto a relojes de dispositivo
            self.dets.append((self.get_clock().now(), msg.header.frame_id,
                              T_from_ros_pose(msg.pose)))

    def _on_js(self, msg):
        with self.lock:
            if self.joints is None:
                self.joints = {}
            self.joints.update(zip(msg.name, msg.position))

    def _auto_start_once(self):
        if not self.running:
            self._launch_worker()

    def _on_start(self, _req, res):
        if self.running:
            res.success, res.message = False, 'ya en ejecución'
        else:
            self._launch_worker()
            res.success, res.message = True, 'calibración iniciada'
        return res

    def _launch_worker(self):
        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------------- helpers
    def _wait(self, fut, timeout):
        ev = threading.Event()
        fut.add_done_callback(lambda _f: ev.set())
        if not ev.wait(timeout):
            return None
        return fut.result()

    def _lookup(self, parent, child, stamp=None, timeout=1.0):
        t = stamp if stamp is not None else Time()
        tf = self.tf_buffer.lookup_transform(parent, child, t, timeout=Duration(seconds=timeout))
        return T_from_ros_transform(tf.transform)

    def _wait_joints(self, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                if self.joints:
                    return dict(self.joints)
            time.sleep(0.1)
        return None

    def _base_request(self):
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name = self.P('move_group')
        r.num_planning_attempts = int(self.P('planning_attempts'))
        r.allowed_planning_time = float(self.P('planning_time'))
        r.max_velocity_scaling_factor = float(self.P('velocity_scaling'))
        r.max_acceleration_scaling_factor = float(self.P('acceleration_scaling'))
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        return goal

    def _move(self, constraints, target_joints=None):
        """Planifica y ejecuta. Con target_joints {nombre: q}, un CONTROL_FAILED que deja el
        robot a <= accept_joint_error_rad del objetivo se acepta: para calibrar no importa
        llegar exacto (se mide la pose real por TF), solo que el robot esté quieto."""
        goal = self._base_request()
        goal.request.goal_constraints = [constraints]
        gh = self._wait(self.move_client.send_goal_async(goal), 10.0)
        if gh is None or not gh.accepted:
            self.get_logger().error('objetivo rechazado por move_group')
            return False
        res = self._wait(gh.get_result_async(), 120.0)
        if res is None:
            self.get_logger().error('timeout esperando ejecución; cancelando')
            gh.cancel_goal_async()
            return False
        code = res.result.error_code.val
        if code == MoveItErrorCodes.CONTROL_FAILED and target_joints:
            time.sleep(0.5)
            with self.lock:
                err = max(abs(self.joints[n] - q) for n, q in target_joints.items())
            if err <= float(self.P('accept_joint_error_rad')):
                self.get_logger().info(
                    f'ejecución abortada por tolerancia ({err:.3f} rad del objetivo); se usa la pose alcanzada')
                return True
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(f'MoveIt error_code={code}')
            return False
        return True

    def _wrist_constraints(self):
        """Limita el enrollado del cable de la cámara: muñeca cerca de la pose inicial."""
        c = Constraints()
        win = np.deg2rad(float(self.P('wrist_window_deg')))
        for name, q0 in getattr(self, 'wrist_home', {}).items():
            c.joint_constraints.append(JointConstraint(
                joint_name=name, position=float(q0), tolerance_above=win, tolerance_below=win,
                weight=1.0))
        return c

    def _ik(self, T_base_eef, joint_names, attempts=3):
        """IK sembrada con la configuración actual (KDL converge a la solución más
        cercana: mínimo giro de muñeca entre poses). None si no hay solución."""
        if not self.ik_client.wait_for_service(timeout_sec=2.0):
            return None
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = self.P('move_group')
        r.ik_link_name = self.eef
        r.avoid_collisions = True
        r.timeout = Duration(seconds=0.2).to_msg()
        r.constraints = self._wrist_constraints()
        r.pose_stamped.header.frame_id = self.base
        fill_ros_pose(r.pose_stamped.pose, T_base_eef)
        with self.lock:
            r.robot_state.joint_state.name = list(self.joints)
            r.robot_state.joint_state.position = [float(v) for v in self.joints.values()]
        for _ in range(attempts):
            res = self._wait(self.ik_client.call_async(req), 5.0)
            if res is not None and res.error_code.val == MoveItErrorCodes.SUCCESS:
                sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
                return [sol[n] for n in joint_names]
        return None

    def _pose_constraints(self, T_base_eef):
        c = self._wrist_constraints()
        pc = PositionConstraint()
        pc.header.frame_id = self.base
        pc.link_name = self.eef
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.001])
        pc.constraint_region.primitives.append(sphere)
        pose = fill_ros_pose(Pose(), T_base_eef)
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id = self.base
        oc.link_name = self.eef
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
        oc.weight = 1.0
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        return c

    @staticmethod
    def _joint_constraints(names, q):
        c = Constraints()
        for n, v in zip(names, q):
            c.joint_constraints.append(JointConstraint(
                joint_name=n, position=float(v), tolerance_above=0.002, tolerance_below=0.002,
                weight=1.0))
        return c

    def _add_table(self):
        if not self.P('table.add'):
            return True
        if not self.scene_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/apply_planning_scene no disponible')
            return False
        co = CollisionObject()
        co.header.frame_id = self.base
        co.id = 'handeye_table'
        th = float(self.P('table.thickness'))
        co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[
            float(self.P('table.size_x')), float(self.P('table.size_y')), th]))
        p = Pose()
        p.position.x = float(self.P('table.center_x'))
        p.position.y = float(self.P('table.center_y'))
        # 1 mm bajo la superficie: link_base apoya sobre la mesa
        p.position.z = float(self.P('table.z')) - th / 2.0 - 0.001
        p.orientation.w = 1.0
        co.primitive_poses.append(p)
        co.operation = CollisionObject.ADD
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(co)
        res = self._wait(self.scene_client.call_async(ApplyPlanningScene.Request(scene=scene)), 5.0)
        return bool(res and res.success)

    def _collect(self, since, n, timeout):
        """Detecciones con stamp >= since."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                got = [d for d in self.dets if d[0] >= since]
            if len(got) >= n:
                return got[:n]
            time.sleep(0.05)
        with self.lock:
            return [d for d in self.dets if d[0] >= since]

    def _capture(self):
        time.sleep(float(self.P('settle_time')))
        since = self.get_clock().now()
        try:
            G0 = self._lookup(self.base, self.eef)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'TF {self.base}->{self.eef}: {e}')
            return None
        dets = self._collect(since, int(self.P('samples_per_pose')), float(self.P('capture_timeout')))
        if len(dets) < max(3, int(self.P('samples_per_pose')) // 2):
            self.get_logger().warn(f'pocas detecciones ({len(dets)}); muestra descartada')
            return None
        G1 = self._lookup(self.base, self.eef)
        if np.linalg.norm(G1[:3, 3] - G0[:3, 3]) * 1000 > self.P('max_robot_motion_mm'):
            self.get_logger().warn('el robot se movió durante la captura; muestra descartada')
            return None
        Cs = [d[2] for d in dets]
        C = mean_T(Cs)
        spread = max(np.linalg.norm(c[:3, 3] - C[:3, 3]) for c in Cs) * 1000
        if spread > self.P('max_sample_spread_mm'):
            self.get_logger().warn(f'dispersión {spread:.2f} mm > umbral (vibración); descartada')
            return None
        return {'T_base_eef': G1, 'T_cam_board': C, 'n_detections': len(Cs),
                'optical_frame': dets[-1][1]}

    def _initial_board_pose(self, X0):
        now = self.get_clock().now()
        since = (now - Duration(seconds=1.0) if now.nanoseconds > 1_000_000_000
                 else Time(clock_type=now.clock_type))
        dets = self._collect(since, 5, 5.0)
        if dets:
            C0 = mean_T([d[2] for d in dets])
            G0 = self._lookup(self.base, self.eef)
            W = G0 @ X0 @ C0
            self.get_logger().info('pose del tablero estimada desde la detección (con X previo)')
            return W, G0 @ X0
        fb = list(self.P('board_pose_base'))
        if self.P('use_board_pose_base') and len(fb) == 6:
            W = T_from_xyz_rpy(fb[:3], fb[3:])
            self.get_logger().info('sin detección: usando board_pose_base')
            # cámara supuesta en la pose actual
            return W, self._lookup(self.base, self.eef) @ X0
        return None, None

    # ---------------------------------------------------------------- worker
    def _worker(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'calibración abortada: {e!r}')
        finally:
            self.running = False

    def _run(self):
        log = self.get_logger()
        capture = bool(self.P('capture'))
        if not self.move_client.wait_for_server(timeout_sec=10.0):
            log.error(f'acción {self.P("move_action")} no disponible (¿move_group lanzado?)')
            return
        js = self._wait_joints()
        if js is None:
            log.error('sin /joint_states')
            return
        try:
            self._lookup(self.base, self.eef, timeout=5.0)
        except Exception as e:  # noqa: BLE001
            log.error(f'TF {self.base}->{self.eef} no disponible: {e}')
            return
        joint_names = sorted(n for n in js if n.startswith('joint'))
        home = [js[n] for n in joint_names]
        self.wrist_home = {n: js[n] for n in ('joint4', 'joint6') if n in js}
        if not self._add_table():
            log.error('no se pudo agregar la mesa a la escena; abortando')
            return

        poses_file = self.P('poses_file')
        targets = []
        if poses_file and os.path.exists(calib_io.expand(poses_file)):
            names, qs = calib_io.load_poses(poses_file)
            joint_names = names
            targets = [('joint', q) for q in qs]
            log.info(f'reproduciendo {len(qs)} poses articulares de {poses_file}')
        else:
            p = self.P('x_prior_xyz_rpy')
            X0 = T_from_xyz_rpy(p[:3], p[3:])
            W, T_cam_now = self._initial_board_pose(X0)
            if W is None:
                log.error('sin detección del tablero y sin board_pose_base; abortando')
                return
            normal = pose_sampler.board_normal_towards(W, T_cam_now[:3, 3])
            vps = pose_sampler.generate_viewpoints(
                W, self.cfg.center, normal, X0, n_poses=int(self.P('n_poses')),
                dist_range=(self.P('distance_min'), self.P('distance_max')),
                tilt_range_deg=(self.P('tilt_min_deg'), self.P('tilt_max_deg')),
                roll_max_deg=self.P('roll_max_deg'), up_hint=-T_cam_now[:3, 1],
                seed=int(self.P('seed')), azimuth_towards=np.zeros(3),
                azimuth_range_deg=float(self.P('azimuth_range_deg')),
                follow_azimuth=bool(self.P('follow_azimuth')))
            zmin = float(self.P('table.z')) + float(self.P('min_eef_height'))
            targets = [('pose', T) for T, _ in vps if T[2, 3] >= zmin]
            log.info(f'{len(targets)}/{len(vps)} poses generadas sobre z>={zmin:.3f} m')

        samples, reached = [], []
        for k, (kind, tgt) in enumerate(targets):
            log.info(f'pose {k + 1}/{len(targets)}')
            q = tgt if kind == 'joint' else self._ik(tgt, joint_names)
            c = (self._joint_constraints(joint_names, q) if q is not None
                 else self._pose_constraints(tgt))
            if not self._move(c, dict(zip(joint_names, q)) if q is not None else None):
                log.warn(f'pose {k + 1} inalcanzable / falló; se omite')
                continue
            time.sleep(0.2)
            with self.lock:
                reached.append([self.joints[n] for n in joint_names])
            if capture:
                s = self._capture()
                if s is not None:
                    s['joints'] = dict(zip(joint_names, reached[-1]))
                    samples.append(s)
                    log.info(f'muestra {len(samples)} OK ({s["n_detections"]} detecciones)')

        log.info('volviendo a la pose inicial')
        self._move(self._joint_constraints(joint_names, home))

        if not (poses_file and os.path.exists(calib_io.expand(poses_file))):
            pf = os.path.join(self.out_dir, f'{self.name}_poses.yaml')
            calib_io.save_poses(pf, joint_names, reached)
            if reached:
                q = np.rad2deg(np.array(reached))
                idx = [joint_names.index(n) for n in ('joint4', 'joint6') if n in joint_names]
                log.info('rango de muñeca recorrido (cable de la cámara): ' + ', '.join(
                    f'{joint_names[i]} {q[:, i].min():.0f}..{q[:, i].max():.0f}°' for i in idx))
            log.info(f'{len(reached)} poses alcanzables guardadas en {pf} '
                     f'(reproducir con poses_file:={pf})')
        if not capture:
            return
        self._solve(samples)

    def _solve(self, samples):
        log = self.get_logger()
        if len(samples) < int(self.P('min_samples')):
            log.error(f'{len(samples)} muestras < min_samples={self.P("min_samples")}; sin resultado')
            return
        optical = samples[-1]['optical_frame']
        root = self.P('camera_root_frame')
        try:
            T_ro = self._lookup(root, optical) if root and root != optical else None
        except Exception as e:  # noqa: BLE001
            log.warn(f'TF {root}->{optical} no disponible ({e}); se publicará eef->óptico')
            T_ro = None
        frames = {'robot_base_frame': self.base, 'robot_effector_frame': self.eef,
                  'camera_root_frame': root if T_ro is not None else optical,
                  'optical_frame': optical}
        sf = os.path.join(self.out_dir, f'{self.name}_samples.yaml')
        calib_io.save_samples(sf, frames, samples, T_ro)
        w, h = self.cfg.size
        pts = np.array([[0, 0, 0], [w, 0, 0], [0, h, 0], [w, h, 0]], float)
        rf = os.path.join(self.out_dir, f'{self.name}.yaml')
        G = [s['T_base_eef'] for s in samples]
        C = [s['T_cam_board'] for s in samples]
        result, _ = calib_io.solve_and_save(rf, self.name, frames, G, C, T_ro,
                                            method=self.P('method'), board_points=pts)
        m = result['metrics']
        log.info(f'RESULTADO -> {rf}\n  muestras usadas {len(result["samples_used"])}/{len(G)}; '
                 f'consistencia tablero: {m["trans_rms_mm"]:.2f} mm RMS (máx {m["trans_max_mm"]:.2f}), '
                 f'{m["rot_rms_deg"]:.3f}° RMS\n  {result["static_transform_publisher"]}')
        if self.P('publish_result'):
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = result['publish']['parent_frame']
            tf.child_frame_id = result['publish']['child_frame']
            fill_ros_transform(tf.transform, calib_io.tf_dict_to_T(result['publish']))
            self.static_tf.sendTransform(tf)


def main():
    rclpy.init()
    node = AutoCalibrator()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
