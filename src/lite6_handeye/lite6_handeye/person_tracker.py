"""Detección de personas en RGB y seguimiento Visual Servoing 6-DOF (Eye-in-Hand).

Combina:
  1. Detector multi-etapa en imagen RGB:
     - Red neuronal ONNX (YOLO26 / YOLOv8, clase COCO 'person') con onnxruntime si se
       configura `onnx_model`; en ese caso es el único detector (sin heurísticas).
     - Detector de peatones HOG de OpenCV (`HOGDescriptor_getDefaultPeopleDetector`).
     - Detector geométrico/colorimétrico de torso + cabeza (humanoide), eficaz a corta
       y media distancia (0.5–3.0 m) cuando la mesa oculta las piernas y en Gazebo.
  2. Reconstrucción 3D del objetivo en `link_base` usando los intrínsecos `K` y la
     transformación calibrada Hand-Eye `X = T_eef_optical`, filtrada con un Kalman de
     velocidad constante (`TargetKalman`) que suaviza y descarta saltos.
  3. Generador de pose 6-DOF (`T_base_eef`) que traslada la cámara en 3D (radio,
     acimut y altura) hacia la persona y orienta el eje óptico (`+z`) directamente al
     objetivo manteniendo el horizonte nivelado (`-y` hacia `+z` de `link_base`).
  4. Cinemática directa e inversa numérica 6-DOF amortiguada (Levenberg-Marquardt)
     del UFactory Lite 6 sembrada en la configuración articular actual.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .pose_sampler import look_at
from .transforms import inv_T, make_T

# Cinemática nominal del UFactory Lite 6 (de lite6_default_kinematics.yaml, URDF oficial)
LITE6_JOINTS = (
    (np.array([0.0, 0.0, 0.2435]), np.array([0.0, 0.0, 0.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([np.pi / 2, -np.pi / 2, np.pi])),
    (np.array([0.2002, 0.0, 0.0]), np.array([-np.pi, 0.0, np.pi / 2])),
    (np.array([0.087, -0.22761, 0.0]), np.array([np.pi / 2, 0.0, 0.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([np.pi / 2, 0.0, 0.0])),
    (np.array([0.0, 0.0625, 0.0]), np.array([-np.pi / 2, 0.0, 0.0])),
)
LITE6_JOINT_LOWER = np.array([-2 * np.pi, -2.61799, -0.061087, -2 * np.pi, -2.1642, -2 * np.pi])
LITE6_JOINT_UPPER = np.array([2 * np.pi, 2.61799, 5.23599, 2 * np.pi, 2.1642, 2 * np.pi])

# Pose inicial de vigilancia: cámara mirando horizontalmente hacia +x de link_base (q3 - q5 = pi/2)
WATCH_JOINTS = [0.0, 0.0, np.pi / 4, 0.0, -np.pi / 4, -np.pi / 2]


@dataclass
class PersonDetection:
    bbox: Tuple[int, int, int, int]        # (x, y, w, h) en píxeles
    track_uv: Tuple[float, float]          # punto de apuntamiento (cabeza/pecho) en píxeles
    head_uv: Tuple[float, float]           # centro estimado de la cabeza en píxeles
    est_distance: float                    # distancia estimada (m)
    score: float                           # confianza [0, 1]
    method: str                            # 'yolo' | 'yolo-pose' | 'hog' | 'humanoid'
    face_kpts: Optional[np.ndarray] = None  # (n, 2) nariz/ojos/orejas visibles (YOLO pose)
    feat: Optional[np.ndarray] = None       # descriptor de apariencia para re-ID (byte_tracker)


def lite6_fk(q) -> np.ndarray:
    """Cinemática directa exacta del Lite 6: q (6 rad) -> T_base_eef (4x4)."""
    q = np.asarray(q, dtype=float).ravel()
    T = np.eye(4)
    for (xyz, rpy), qi in zip(LITE6_JOINTS, q):
        R0 = Rotation.from_euler('xyz', rpy).as_matrix()
        Rz = Rotation.from_rotvec([0.0, 0.0, float(qi)]).as_matrix()
        Ti = np.eye(4)
        Ti[:3, :3] = R0 @ Rz
        Ti[:3, 3] = xyz
        T = T @ Ti
    return T


def lite6_ik_6dof(T_target: np.ndarray, q_seed, max_iter: int = 50,
                  tol_pos: float = 2e-4, tol_rot: float = 5e-4,
                  damping: float = 1.5e-2) -> Tuple[np.ndarray, bool]:
    """IK numérica 6-DOF (Levenberg-Marquardt) sembrada en q_seed.

    Devuelve (q_sol, exito). Garantiza continuidad articular sin saltos de muñeca.
    """
    q = np.clip(np.asarray(q_seed, dtype=float).ravel().copy(),
                LITE6_JOINT_LOWER, LITE6_JOINT_UPPER)
    eps = 1e-6
    lam2 = damping ** 2
    I6 = np.eye(6)
    for _ in range(max_iter):
        T = lite6_fk(q)
        err_p = T_target[:3, 3] - T[:3, 3]
        err_r = T[:3, :3] @ Rotation.from_matrix(T[:3, :3].T @ T_target[:3, :3]).as_rotvec()
        if np.linalg.norm(err_p) <= tol_pos and np.linalg.norm(err_r) <= tol_rot:
            return q, True
        err = np.concatenate([err_p, err_r])
        J = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = eps
            Ti = lite6_fk(q + dq)
            J[:3, i] = (Ti[:3, 3] - T[:3, 3]) / eps
            J[3:, i] = (T[:3, :3] @ Rotation.from_matrix(T[:3, :3].T @ Ti[:3, :3]).as_rotvec()) / eps
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * I6, err)
        step_norm = np.max(np.abs(dq))
        if step_norm > 0.35:
            dq *= 0.35 / step_norm
        q = np.clip(q + dq, LITE6_JOINT_LOWER, LITE6_JOINT_UPPER)
    T = lite6_fk(q)
    ok = (np.linalg.norm(T_target[:3, 3] - T[:3, 3]) <= 2e-3
          and np.linalg.norm(Rotation.from_matrix(T[:3, :3].T @ T_target[:3, :3]).as_rotvec()) <= 5e-3)
    return q, bool(ok)


def ray_to_target_3d(u: float, v: float, K: np.ndarray, T_base_eef: np.ndarray,
                     X_eef_opt: np.ndarray, distance: float) -> np.ndarray:
    """Proyecta el píxel (u, v) a un punto 3D en `link_base` usando K y X calibrada."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    ray_cam /= np.linalg.norm(ray_cam)
    p_cam = ray_cam * float(distance)
    T_base_cam = T_base_eef @ X_eef_opt
    return (T_base_cam @ np.append(p_cam, 1.0))[:3]


class TargetKalman:
    """Filtro de Kalman 3D de velocidad constante para el objetivo en `link_base`.

    Estado [p, v] (6). Ruido de proceso: aceleración blanca de desviación `accel_std`.
    Ruido de medición anisótropo: `meas_lat_std` perpendicular al rayo de la cámara y
    `meas_depth_std` a lo largo del rayo (la distancia se estima por el alto del bbox).
    Las mediciones con distancia de Mahalanobis > `gate_sigma` se descartan (falsas
    detecciones); tras `max_rejects` descartes seguidos el filtro se reinicia.
    """

    def __init__(self, accel_std: float = 0.5, meas_lat_std: float = 0.03,
                 meas_depth_std: float = 0.15, gate_sigma: float = 3.5, max_rejects: int = 5):
        self.accel_std = float(accel_std)
        self.meas_lat_std = float(meas_lat_std)
        self.meas_depth_std = float(meas_depth_std)
        self.gate_sigma = float(gate_sigma)
        self.max_rejects = int(max_rejects)
        self.reset()

    def reset(self):
        self.x = None              # [px py pz vx vy vz]
        self.P = None
        self.t = None
        self.rejects = 0

    @property
    def position(self) -> Optional[np.ndarray]:
        return None if self.x is None else self.x[:3].copy()

    def _R(self, ray_dir: Optional[np.ndarray]) -> np.ndarray:
        lat2, dep2 = self.meas_lat_std ** 2, self.meas_depth_std ** 2
        if ray_dir is None:
            return np.eye(3) * dep2
        r = np.asarray(ray_dir, dtype=float).ravel()
        r = r / np.linalg.norm(r)
        return lat2 * np.eye(3) + (dep2 - lat2) * np.outer(r, r)

    def _init(self, z: np.ndarray, t: float, R: np.ndarray):
        self.x = np.concatenate([z, np.zeros(3)])
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = R
        self.P[3:, 3:] = np.eye(3) * 0.25          # (0.5 m/s)^2: velocidad inicial desconocida
        self.t = t
        self.rejects = 0

    def predict(self, t: float):
        dt = max(0.0, float(t) - self.t)
        if dt == 0.0:
            return
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.accel_std ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * q * dt ** 4 / 4.0
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * q * dt ** 3 / 2.0
        Q[3:, 3:] = np.eye(3) * q * dt ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = float(t)

    def update(self, z, t: float, ray_dir=None) -> bool:
        """Incorpora la medición `z` (3D, link_base) en el instante `t` (s). False si se descartó."""
        z = np.asarray(z, dtype=float).ravel()
        R = self._R(ray_dir)
        if self.x is None:
            self._init(z, t, R)
            return True
        self.predict(t)
        y = z - self.x[:3]
        S = self.P[:3, :3] + R
        S_inv = np.linalg.inv(S)
        if float(y @ S_inv @ y) > self.gate_sigma ** 2:
            self.rejects += 1
            if self.rejects >= self.max_rejects:
                self._init(z, t, R)
                return True
            return False
        Kg = self.P[:, :3] @ S_inv
        self.x = self.x + Kg @ y
        self.P = self.P - Kg @ self.P[:3, :]       # (I - K H) P, con H = [I 0]
        self.rejects = 0
        return True


def compute_tracking_pose_6dof(
    target_base: np.ndarray,
    X_eef_opt: np.ndarray,
    cam_radius_nom: float = 0.30,
    cam_radius_gain: float = 0.05,
    cam_radius_range: Tuple[float, float] = (0.24, 0.34),
    cam_z_nom: float = 0.38,
    cam_z_gain: float = 0.28,
    cam_z_range: Tuple[float, float] = (0.28, 0.46),
    max_azimuth_deg: float = 75.0,
    az_ref: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula (T_base_eef, T_base_cam) en 6-DOF para seguir a `target_base` en `link_base`.

    La cámara se traslada en 3D (acimut, avance radial y altura z) hacia la persona y
    orienta su eje óptico (+z) directamente hacia `target_base` con el horizonte nivelado.
    `az_ref` (rad, p. ej. el acimut actual de la cámara) elige la vuelta más corta al cruzar ±180°.
    """
    target = np.asarray(target_base, dtype=float).ravel()
    d_xy = float(np.linalg.norm(target[:2]))
    az = float(np.arctan2(target[1], target[0]))      # 360°: la base (J1) gira hacia la persona
    if az_ref is not None:                            # desenrollar: no dar la vuelta larga en ±180°
        az = az_ref + float(np.arctan2(np.sin(az - az_ref), np.cos(az - az_ref)))
    max_az = np.deg2rad(max_azimuth_deg)
    az_clamped = float(np.clip(az, -max_az, max_az))

    r_xy = float(np.clip(cam_radius_nom + cam_radius_gain * (d_xy - 1.25),
                         cam_radius_range[0], cam_radius_range[1]))
    z_cam = float(np.clip(cam_z_nom + cam_z_gain * (target[2] - cam_z_nom),
                          cam_z_range[0], cam_z_range[1]))

    cam_pos = np.array([r_xy * np.cos(az_clamped), r_xy * np.sin(az_clamped), z_cam])
    # Si el objetivo está demasiado cerca de cam_pos, garantiza vector de vista válido
    if np.linalg.norm(target - cam_pos) < 0.15:
        target = cam_pos + np.array([np.cos(az_clamped), np.sin(az_clamped), 0.0]) * 0.5

    R_cam = look_at(cam_pos, target, np.array([0.0, 0.0, -1.0]))
    T_base_cam = make_T(R_cam, cam_pos)
    T_base_eef = T_base_cam @ inv_T(X_eef_opt)
    return T_base_eef, T_base_cam


class PersonDetector:
    """Detector ligero de personas en RGB (DNN ONNX opcional + HOG + torso/cabeza humanoide)."""

    def __init__(self, onnx_model: str = '', ref_torso_height_m: float = 0.52,
                 dist_range: Tuple[float, float] = (0.6, 3.0), min_score: float = 0.25):
        self.ref_h = float(ref_torso_height_m)
        self.dist_range = (float(dist_range[0]), float(dist_range[1]))
        self.min_score = float(min_score)
        self.net = None            # sesión onnxruntime (YOLO26/YOLOv8, salida (1, 4+C, N))
        self.net_size = 640
        if onnx_model:
            import onnxruntime as ort  # OpenCV 4.6 no importa la atención de YOLO26
            if hasattr(ort, 'preload_dlls'):
                ort.preload_dlls()     # CUDA/cuDNN de los wheels nvidia-* (onnxruntime-gpu[cuda,cudnn])
            providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                         if p in ort.get_available_providers()]
            self.net = ort.InferenceSession(os.path.expanduser(onnx_model), providers=providers)
            self.net_provider = self.net.get_providers()[0]
            self.net_input = self.net.get_inputs()[0].name
            self.net_size = int(self.net.get_inputs()[0].shape[2])
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _estimate_distance(self, bbox_h: int, K: Optional[np.ndarray], img_h: int) -> float:
        fy = float(K[1, 1]) if K is not None else img_h * 0.85
        d = fy * self.ref_h / max(float(bbox_h), 12.0)
        return float(np.clip(d, self.dist_range[0], self.dist_range[1]))

    def _detect_humanoid(self, bgr: np.ndarray, K: Optional[np.ndarray]) -> list[PersonDetection]:
        """Detecta silueta humanoide (cabeza tono piel + torso superior o torso prominente)."""
        h_img, w_img = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

        # 1. Máscara de piel / cabeza (combina YCrCb + HSV para iluminación real y sintética)
        skin_ycrcb = cv2.inRange(ycrcb, (40, 132, 75), (255, 178, 132))
        skin_hsv = cv2.inRange(hsv, (0, 25, 60), (28, 210, 255))
        skin = cv2.bitwise_or(skin_ycrcb, skin_hsv)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k5)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k5)

        # 2. Máscara de torso con saturación cromática (azul/verde/rojo/etc. sobre fondo neutro)
        torso_cool = cv2.inRange(hsv, (35, 55, 35), (170, 255, 255))
        torso_red = cv2.inRange(hsv, (170, 65, 40), (180, 255, 255))
        torso_mask = cv2.bitwise_or(torso_cool, torso_red)
        torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_OPEN, k5)
        torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

        heads = []
        cnts_head, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_head:
            area = cv2.contourArea(c)
            if area < 120 or area > 0.15 * h_img * w_img:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(float(h), 1.0)
            if 0.45 <= aspect <= 1.8:
                heads.append((x, y, w, h, area))
        if heads:
            heads.sort(key=lambda item: item[4], reverse=True)
            max_head_area = heads[0][4]
            # Descarta manos u otras regiones pequeñas de piel cuando la cabeza está visible
            heads = [h for h in heads if h[4] >= 0.38 * max_head_area]

        torsos = []
        cnts_torso, _ = cv2.findContours(torso_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_torso:
            area = cv2.contourArea(c)
            if area < 400 or area > 0.45 * h_img * w_img:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(float(h), 1.0)
            if 0.35 <= aspect <= 2.6:
                torsos.append((x, y, w, h, area))

        dets: list[PersonDetection] = []
        used_torsos = set()
        for hx, hy, hw, hh, _ in heads:
            hcx, hcy = hx + 0.5 * hw, hy + 0.5 * hh
            best_t = None
            for idx, (tx, ty, tw, th, _) in enumerate(torsos):
                tcx = tx + 0.5 * tw
                # La cabeza debe estar en la parte superior del torso (incluso con brazos alzados)
                if abs(hcx - tcx) <= 0.95 * tw and -1.3 * hh <= (ty - (hy + hh)) <= 1.6 * hh:
                    best_t = (idx, tx, ty, tw, th)
                    break
            if best_t is not None:
                idx, tx, ty, tw, th = best_t
                used_torsos.add(idx)
                x0 = int(min(hx, tx))
                y0 = int(min(hy, ty))
                x1 = int(max(hx + hw, tx + tw))
                y1 = int(max(hy + hh, ty + th))
                bw, bh = x1 - x0, y1 - y0
                track_u = 0.65 * hcx + 0.35 * (tx + 0.5 * tw)
                track_v = float(hy + 0.75 * hh)
                eff_bh = int(min(float(bh), 3.6 * float(max(hh, hw))))
                dist = self._estimate_distance(eff_bh, K, h_img)
                dets.append(PersonDetection(
                    bbox=(x0, y0, bw, bh),
                    track_uv=(float(track_u), track_v),
                    head_uv=(float(hcx), float(hcy)),
                    est_distance=dist,
                    score=0.92,
                    method='humanoid',
                ))
            elif hw >= 18 and hh >= 18:
                # Rostro/cabeza visible de cerca: infiere caja de torso debajo
                bw = int(2.4 * hw)
                bh = int(3.2 * hh)
                x0 = int(np.clip(hcx - 0.5 * bw, 0, w_img - 1))
                y0 = int(hy)
                bw = min(bw, w_img - x0)
                bh = min(bh, h_img - y0)
                dist = self._estimate_distance(bh, K, h_img)
                dets.append(PersonDetection(
                    bbox=(x0, y0, bw, bh),
                    track_uv=(float(hcx), float(hy + 0.8 * hh)),
                    head_uv=(float(hcx), float(hcy)),
                    est_distance=dist,
                    score=0.65,
                    method='humanoid',
                ))

        for idx, (tx, ty, tw, th, _) in enumerate(torsos):
            if idx in used_torsos:
                continue
            tcx = tx + 0.5 * tw
            head_y = max(0.0, ty - 0.35 * th)
            y0 = int(head_y)
            bh = int(ty + th - y0)
            dist = self._estimate_distance(bh, K, h_img)
            dets.append(PersonDetection(
                bbox=(int(tx), y0, int(tw), bh),
                track_uv=(float(tcx), float(ty + 0.25 * th)),
                head_uv=(float(tcx), float(ty - 0.15 * th)),
                est_distance=dist,
                score=0.70,
                method='humanoid',
            ))
        return dets

    def _detect_hog(self, bgr: np.ndarray, K: Optional[np.ndarray]) -> list[PersonDetection]:
        h_img, w_img = bgr.shape[:2]
        scale = 0.5 if w_img >= 960 else 1.0
        small = cv2.resize(bgr, (0, 0), fx=scale, fy=scale) if scale != 1.0 else bgr
        rects, weights = self.hog.detectMultiScale(
            small, hitThreshold=0.1, winStride=(8, 8), padding=(16, 16), scale=1.06)
        out: list[PersonDetection] = []
        if rects is None or len(rects) == 0:
            return out
        for (x, y, w, h), weight in zip(rects, np.asarray(weights).ravel()):
            x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
            score = float(np.clip(0.4 + 0.3 * float(weight), 0.0, 0.99))
            if score < self.min_score:
                continue
            cx = x + 0.5 * w
            head_v = y + 0.18 * h
            track_v = y + 0.28 * h
            dist = self._estimate_distance(int(0.55 * h), K, h_img)
            out.append(PersonDetection(
                bbox=(x, y, w, h),
                track_uv=(float(cx), float(track_v)),
                head_uv=(float(cx), float(head_v)),
                est_distance=dist,
                score=score,
                method='hog',
            ))
        return out

    def _detect_dnn(self, bgr: np.ndarray, K: Optional[np.ndarray],
                    thresh: Optional[float] = None) -> list[PersonDetection]:
        """YOLO + NMS, clase 'person'. Salida cruda (1, 4+C, N):
        detección (C = 80 clases COCO) o pose (C = 1 + 17x3 puntos clave COCO). Con pose, el punto
        de apuntamiento es la cabeza (media ponderada de nariz, ojos y orejas visibles)."""
        h_img, w_img = bgr.shape[:2]
        n = self.net_size
        r = min(n / w_img, n / h_img)                       # letterbox: conserva proporción
        nw, nh = int(round(w_img * r)), int(round(h_img * r))
        px, py = (n - nw) // 2, (n - nh) // 2
        canvas = np.full((n, n, 3), 114, dtype=np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, swapRB=True)
        pred = self.net.run(None, {self.net_input: blob})[0][0].T     # (N, 4+C)
        thresh = self.min_score if thresh is None else float(thresh)
        conf = pred[:, 4]                                             # clase 0 = person (COCO)
        keep = conf >= thresh
        if not np.any(keep):
            return []
        pose = pred.shape[1] == 56
        cxcywh, conf = pred[keep, :4], conf[keep]
        if pose:
            kp = pred[keep, 5:].reshape(-1, 17, 3).copy()
            kp[:, :, 0] = (kp[:, :, 0] - px) / r
            kp[:, :, 1] = (kp[:, :, 1] - py) / r
        boxes = np.column_stack([(cxcywh[:, 0] - cxcywh[:, 2] / 2 - px) / r,
                                 (cxcywh[:, 1] - cxcywh[:, 3] / 2 - py) / r,
                                 cxcywh[:, 2] / r, cxcywh[:, 3] / r])
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), thresh, 0.45)
        out: list[PersonDetection] = []
        fy = float(K[1, 1]) if K is not None else h_img * 0.85
        for i in np.asarray(idx).ravel():
            x, y, w, h = boxes[i]
            x, y = max(0.0, x), max(0.0, y)
            w, h = min(w, w_img - x), min(h, h_img - y)
            # cuerpo completo visible: alto ~1.70 m; cortado abajo: ancho de hombros ~0.45 m
            cut = y + h >= 0.98 * h_img
            d = fy * (0.45 / max(w, 12.0) if cut else 1.70 / max(h, 12.0))
            head_h = min(h, 2.2 * w) if cut else h           # alto aprox. del tramo cabeza-torso
            head = (float(x + w / 2), float(y + 0.10 * head_h))
            method, face = 'yolo', None
            if pose:
                f = kp[i, :5]                                 # nariz, ojos, orejas
                vis = f[:, 2] >= 0.5
                if np.any(vis):
                    wgt = f[vis, 2]
                    head = (float(np.average(f[vis, 0], weights=wgt)),
                            float(np.average(f[vis, 1], weights=wgt)))
                    method, face = 'yolo-pose', f[vis, :2]
                sh = kp[i, 5:7]                               # hombros: escala si el cuerpo está cortado
                if cut and np.all(sh[:, 2] >= 0.5):
                    sw = float(np.linalg.norm(sh[0, :2] - sh[1, :2]))
                    if sw > 0.6 * w:                          # de frente: ancho biacromial ~0.38 m
                        d = fy * 0.38 / sw
            out.append(PersonDetection(
                bbox=(int(x), int(y), int(w), int(h)),
                track_uv=head,
                head_uv=head,
                est_distance=float(np.clip(d, self.dist_range[0], self.dist_range[1])),
                score=float(conf[i]),
                method=method,
                face_kpts=face,
            ))
        return out

    def detect_all(self, bgr: np.ndarray, K: Optional[np.ndarray] = None,
                   thresh: float = 0.1) -> list[PersonDetection]:
        """Todas las personas (YOLO) con puntaje >= `thresh`, para el tracker de IDs."""
        if self.net is None:
            return self._detect_humanoid(bgr, K) or self._detect_hog(bgr, K)
        return self._detect_dnn(bgr, K, thresh)

    def detect(self, bgr: np.ndarray, K: Optional[np.ndarray] = None,
               prev_uv: Optional[Tuple[float, float]] = None) -> Optional[PersonDetection]:
        """Devuelve la mejor detección de persona en la imagen BGR (o None)."""
        if self.net is not None:
            candidates = self._detect_dnn(bgr, K)
        else:
            candidates = self._detect_humanoid(bgr, K)
            if not candidates:
                candidates = self._detect_hog(bgr, K)
        if not candidates:
            return None
        if prev_uv is not None and len(candidates) > 1:
            pu, pv = prev_uv
            diag = float(np.hypot(bgr.shape[1], bgr.shape[0]))
            return max(
                candidates,
                key=lambda d: d.score - 0.35 * (np.hypot(d.track_uv[0] - pu, d.track_uv[1] - pv) / diag)
            )
        return max(candidates, key=lambda d: d.score)


def draw_debug(bgr: np.ndarray, det: Optional[PersonDetection],
               target_base: Optional[np.ndarray] = None) -> np.ndarray:
    """Dibuja recuadro, centro óptico, punto de seguimiento y coordenadas 3D."""
    out = bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.drawMarker(out, (cx, cy), (180, 180, 180), cv2.MARKER_CROSS, 24, 1)

    if det is None:
        cv2.putText(out, 'searching person...', (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
        return out

    x, y, bw, bh = det.bbox
    u, v = int(round(det.track_uv[0])), int(round(det.track_uv[1]))
    hu, hv = int(round(det.head_uv[0])), int(round(det.head_uv[1]))
    cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 230, 60), 2)
    cv2.circle(out, (hu, hv), 6, (255, 200, 0), -1)
    if det.face_kpts is not None:
        for p in det.face_kpts:
            cv2.circle(out, (int(p[0]), int(p[1])), 3, (255, 0, 255), -1)
    cv2.drawMarker(out, (u, v), (0, 60, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    cv2.arrowedLine(out, (cx, cy), (u, v), (0, 200, 255), 2, tipLength=0.15)

    err_px = np.hypot(det.track_uv[0] - cx, det.track_uv[1] - cy)
    txt = f'{det.method} {det.score:.2f}  d={det.est_distance:.2f}m  err={err_px:.0f}px'
    if target_base is not None:
        txt += f'  base=({target_base[0]:.2f},{target_base[1]:+.2f},{target_base[2]:.2f})m'
    cv2.putText(out, txt, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 80), 2, cv2.LINE_AA)
    return out
