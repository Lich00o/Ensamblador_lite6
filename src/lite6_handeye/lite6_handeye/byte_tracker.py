"""ByteTrack mínimo (Zhang et al., 2022) para asignar un ID estable a cada persona.

Cada track tiene un Kalman de velocidad constante sobre la caja (cx, cy, w, h). En cada cuadro:
  1. Predicción de todos los tracks.
  2. Asociación (húngaro sobre 1 - IoU) de las detecciones de puntaje ALTO con los tracks
     activos y perdidos.
  3. Asociación de las detecciones de puntaje BAJO con los tracks activos que quedaron sin
     pareja (recupera personas parcialmente ocluidas sin crear IDs nuevos).
  4. Tracks sin pareja pasan a perdidos; se eliminan tras `track_buffer` cuadros.
  5. Detecciones altas sin pareja (>= `new_track_thresh`) crean un track nuevo.
Mismo esquema que el `bytetrack.yaml` de Ultralytics usado en tello_env, sin dependencias
pesadas (numpy + scipy).

Re-identificación (galería de IDs, idea de los trackers con apariencia tipo DeepSORT): IoU solo
recupera a quien reaparece cerca de donde se perdió. Para que una persona que sale del cuadro (o
queda tapada varios segundos) vuelva con su mismo ID, cada track guarda un descriptor de apariencia
(embedding de un modelo ReID, `ReIDEmbedder`: OSNet-AIN x1.0 MSMT17 en ONNX; sin modelo, histogramas
HSV de torso y piernas, `appearance_feature`, que en la prueba real no separaron a dos personas:
similitud 0.70–0.81 entre distintas). Al expirar, el track pasa a una galería que dura
`reid_buffer` cuadros. Un track nuevo queda "en prueba" sus primeros `reid_probation` cuadros: si
su descriptor coincide con un ID de la galería o con un track perdido durante `reid_confirm`
cuadros seguidos (con margen sobre el segundo candidato), toma ese ID.

Compensación del movimiento de cámara (idea de BoT-SORT): la cámara va en el brazo, así que al
girar el robot todas las cajas se desplazan y la asociación por IoU fallaría. BoT-SORT estima ese
movimiento con flujo óptico; aquí se conoce exacto por cinemática: `apply_homography(H)` mueve
las predicciones con H = K R K^-1 (rotación de la cámara entre cuadros) antes de asociar.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cajas (x, y, w, h): a (N, 4), b (M, 4) -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def _match(iou: np.ndarray, min_iou: float):
    if iou.size == 0:
        return [], list(range(iou.shape[0])), list(range(iou.shape[1]))
    r, c = linear_sum_assignment(1.0 - iou)
    pairs = [(i, j) for i, j in zip(r, c) if iou[i, j] >= min_iou]
    mi = {i for i, _ in pairs}
    mj = {j for _, j in pairs}
    return (pairs, [i for i in range(iou.shape[0]) if i not in mi],
            [j for j in range(iou.shape[1]) if j not in mj])


_H_BINS, _S_BINS, _V_BINS = 12, 4, 4


def appearance_feature(bgr: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Descriptor de ropa: histograma H-S (píxeles no oscuros) + histograma V, de torso y piernas.

    Cada parte normalizada a suma 1; se compara con `appearance_similarity` (Bhattacharyya).
    None si la caja es muy chica o cae fuera de la imagen.
    """
    H, W = bgr.shape[:2]
    x, y, w, h = (float(v) for v in bbox)
    if w < 15 or h < 40:
        return None
    x0, x1 = int(max(0, x + 0.2 * w)), int(min(W, x + 0.8 * w))     # centro: menos fondo
    parts = []
    for f0, f1 in ((0.15, 0.5), (0.5, 0.9)):                         # torso, piernas
        y0, y1 = int(max(0, y + f0 * h)), int(min(H, y + f1 * h))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return None
        hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        mask = (hsv[..., 2] > 40).astype(np.uint8)                   # tono no confiable en lo oscuro
        hs = cv2.calcHist([hsv], [0, 1], mask, [_H_BINS, _S_BINS], [0, 180, 0, 256]).ravel()
        v = cv2.calcHist([hsv], [2], None, [_V_BINS], [0, 256]).ravel()
        part = np.concatenate([hs, 0.5 * hs.sum() / max(v.sum(), 1.0) * v])
        parts.append(part / max(part.sum(), 1e-9))
    return np.concatenate(parts)


def appearance_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Coeficiente de Bhattacharyya promedio de las dos partes: 1 = igual, 0 = disjunto."""
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))) / 2.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno de embeddings ReID (no hace falta que vengan normalizados)."""
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


class ReIDEmbedder:
    """Embedding de apariencia por persona con un modelo ReID ONNX (entrada (N, 3, 256, 128) RGB
    normalizada ImageNet, salida (N, D)). Probado con OSNet-AIN x1.0 MSMT17 (mprib/osnet-ain-x1-0-msmt17-onnx)."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, onnx_model: str):
        import os
        import onnxruntime as ort
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                     if p in ort.get_available_providers()]
        self.net = ort.InferenceSession(os.path.expanduser(onnx_model), providers=providers)
        self.provider = self.net.get_providers()[0]
        self.input = self.net.get_inputs()[0].name
        _, _, self.h, self.w = self.net.get_inputs()[0].shape

    def __call__(self, bgr: np.ndarray, bboxes) -> List[Optional[np.ndarray]]:
        H, W = bgr.shape[:2]
        crops, idx = [], []
        for k, (x, y, w, h) in enumerate(bboxes):
            x0, y0 = int(max(0, x)), int(max(0, y))
            x1, y1 = int(min(W, x + w)), int(min(H, y + h))
            if x1 - x0 < 10 or y1 - y0 < 20:
                continue
            rgb = cv2.cvtColor(cv2.resize(bgr[y0:y1, x0:x1], (self.w, self.h)), cv2.COLOR_BGR2RGB)
            crops.append(((rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1))
            idx.append(k)
        out: List[Optional[np.ndarray]] = [None] * len(bboxes)
        if crops:
            emb = self.net.run(None, {self.input: np.stack(crops)})[0]
            for k, e in zip(idx, emb):
                out[k] = e / max(np.linalg.norm(e), 1e-9)
        return out


class Track:
    _std_pos, _std_vel = 1.0 / 20, 1.0 / 160      # ruido relativo a la altura (como ByteTrack)

    def __init__(self, tid: int, det):
        self.id = tid
        self.det = det                             # última detección asociada (PersonDetection)
        x, y, w, h = (float(v) for v in det.bbox)
        self.x = np.array([x + w / 2, y + h / 2, w, h, 0, 0, 0, 0], dtype=float)
        s = max(h, 1.0)
        self.P = np.diag(np.square([2 * self._std_pos * s] * 4 + [10 * self._std_vel * s] * 4))
        self.hits, self.time_since_update, self.age = 1, 0, 0
        self.feat = getattr(det, 'feat', None)     # descriptor de apariencia (promedio móvil)
        self.probation = True                      # recién creado: puede heredar un ID de la galería
        self.vote = (None, 0)                      # (ID candidato, cuadros seguidos coincidiendo)

    def update_feat(self, det, alpha: float = 0.1):
        f = getattr(det, 'feat', None)
        if f is not None:
            self.feat = f.copy() if self.feat is None else (1 - alpha) * self.feat + alpha * f

    @property
    def bbox(self) -> tuple:
        cx, cy, w, h = self.x[:4]
        return (cx - w / 2, cy - h / 2, w, h)

    @property
    def visible(self) -> bool:
        return self.time_since_update == 0

    def predict(self):
        F = np.eye(8)
        F[:4, 4:] = np.eye(4)
        s = max(self.x[3], 1.0)
        Q = np.diag(np.square([self._std_pos * s] * 4 + [self._std_vel * s] * 4))
        self.x = F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        self.time_since_update += 1

    def update(self, det):
        x, y, w, h = (float(v) for v in det.bbox)
        z = np.array([x + w / 2, y + h / 2, w, h])
        s = max(h, 1.0)
        R = np.diag(np.square([self._std_pos * s] * 4))
        S = self.P[:4, :4] + R
        K = self.P[:, :4] @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x[:4])
        self.P = self.P - K @ self.P[:4, :]
        self.det = det
        self.hits += 1
        self.time_since_update = 0


class ByteTracker:
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6, match_iou: float = 0.2, low_match_iou: float = 0.5,
                 track_buffer: int = 30, reid: bool = True, reid_buffer: int = 3600,
                 reid_thresh: float = 0.8, reid_margin: float = 0.05, reid_probation: int = 15,
                 reid_min_hits: int = 5, gallery_size: int = 50, reid_confirm: int = 3,
                 similarity=appearance_similarity):
        self.high, self.low, self.new = float(high_thresh), float(low_thresh), float(new_track_thresh)
        self.match_iou, self.low_match_iou = float(match_iou), float(low_match_iou)
        self.buffer = int(track_buffer)
        self.reid, self.reid_buffer = bool(reid), int(reid_buffer)
        self.reid_thresh, self.reid_margin = float(reid_thresh), float(reid_margin)
        self.reid_probation, self.reid_min_hits = int(reid_probation), int(reid_min_hits)
        self.gallery_size, self.reid_confirm = int(gallery_size), int(reid_confirm)
        self.similarity = similarity
        self.tracks: List[Track] = []
        self.gallery: Dict[int, tuple] = {}        # ID expirado -> (descriptor, cuadro en que expiró)
        self.aliases: Dict[int, int] = {}          # renombres del último cuadro: ID en prueba -> ID recuperado
        self._next_id = 1
        self._frame = 0

    def reset(self):
        self.tracks, self.gallery, self.aliases, self._next_id, self._frame = [], {}, {}, 1, 0

    def known(self, tid: Optional[int]) -> bool:
        """El ID sigue vigente (track activo/perdido) o en la galería (puede volver)."""
        return tid is not None and (self.get(tid) is not None or tid in self.gallery)

    def apply_homography(self, H: np.ndarray):
        """Lleva los tracks al cuadro actual según el movimiento de la cámara (H: 3x3, píxeles)."""
        for t in self.tracks:
            c = H @ np.array([t.x[0], t.x[1], 1.0])
            if abs(c[2]) > 1e-9:
                t.x[0], t.x[1] = c[0] / c[2], c[1] / c[2]

    def get(self, tid: Optional[int]) -> Optional[Track]:
        return next((t for t in self.tracks if t.id == tid), None)

    def update(self, dets: list) -> List[Track]:
        """Incorpora las detecciones del cuadro; devuelve todos los tracks vigentes."""
        self._frame += 1
        self.aliases = {}
        for t in self.tracks:
            t.predict()
        high = [d for d in dets if d.score >= self.high]
        low = [d for d in dets if self.low <= d.score < self.high]
        boxes = lambda ds: np.array([d.bbox for d in ds], dtype=float).reshape(-1, 4)  # noqa: E731

        # 1.ª etapa: altas vs todos los tracks (activos y perdidos)
        tb = np.array([t.bbox for t in self.tracks], dtype=float).reshape(-1, 4)
        pairs, un_t, un_h = _match(iou_matrix(tb, boxes(high)), self.match_iou)
        hb = boxes(high)
        crowd = iou_matrix(hb, hb) - np.eye(len(high))       # solapamiento con otras personas
        for i, j in pairs:
            self.tracks[i].update(high[j])
            if crowd.shape[1] == 0 or crowd[j].max() < 0.1:  # sin oclusión: no mezclar ropa ajena
                self.tracks[i].update_feat(high[j])
        # 2.ª etapa: bajas vs tracks que venían visibles y quedaron sin pareja
        rest = [i for i in un_t if self.tracks[i].time_since_update == 1]
        tb2 = np.array([self.tracks[i].bbox for i in rest], dtype=float).reshape(-1, 4)
        pairs2, _, _ = _match(iou_matrix(tb2, boxes(low)), self.low_match_iou)
        for i, j in pairs2:
            self.tracks[rest[i]].update(low[j])
        # tracks nuevos y limpieza
        for j in un_h:
            if high[j].score >= self.new:
                self.tracks.append(Track(self._next_id, high[j]))
                self._next_id += 1
        if self.reid:
            self._reidentify()
        keep = []
        for t in self.tracks:
            if t.time_since_update <= self.buffer:
                keep.append(t)
            elif self.reid and t.feat is not None and t.hits >= self.reid_min_hits:
                self.gallery[t.id] = (t.feat, self._frame)
        self.tracks = keep
        self.gallery = {k: v for k, v in self.gallery.items() if self._frame - v[1] <= self.reid_buffer}
        if len(self.gallery) > self.gallery_size:             # quedan los más recientes
            old = sorted(self.gallery, key=lambda k: self.gallery[k][1])
            for k in old[:len(self.gallery) - self.gallery_size]:
                del self.gallery[k]
        return list(self.tracks)

    def _reidentify(self):
        """Tracks en prueba visibles vs galería + tracks perdidos; el que coincide hereda el ID."""
        for t in self.tracks:
            if t.probation and t.age >= self.reid_probation:
                t.probation = False
        new = [t for t in self.tracks if t.probation and t.visible and t.feat is not None]
        lost = [t for t in self.tracks if not t.probation and not t.visible and t.feat is not None]
        cands = [(k, f) for k, (f, _) in self.gallery.items()] + [(t.id, t.feat) for t in lost]
        if not new or not cands:
            return
        sim = np.array([[self.similarity(n.feat, f) for _, f in cands] for n in new])
        r, c = linear_sum_assignment(-sim)
        drop = set()
        assigned = dict(zip(r, c))
        for i, n in enumerate(new):
            j = assigned.get(i)
            if j is None:
                n.vote = (None, 0)
                continue
            row = np.delete(sim[i], j)
            col = np.delete(sim[:, j], i)
            second = max(row.max() if row.size else -1.0, col.max() if col.size else -1.0)
            if sim[i, j] < self.reid_thresh or sim[i, j] - second < self.reid_margin:
                n.vote = (None, 0)                            # poco parecido o ambiguo
                continue
            old_id, f = cands[j]
            n.vote = (old_id, n.vote[1] + 1 if n.vote[0] == old_id else 1)
            if n.vote[1] < self.reid_confirm:                 # esperar cuadros seguidos coincidiendo
                continue
            self.aliases[n.id] = old_id
            n.id, n.probation = old_id, False
            n.feat = 0.5 * (n.feat + f)
            if old_id in self.gallery:
                del self.gallery[old_id]
            else:
                drop.add(old_id)
        if drop:
            self.tracks = [t for t in self.tracks if t.visible or t.id not in drop]


def select_at(tracks: List[Track], u: float, v: float) -> Optional[Track]:
    """Track visible cuya caja contiene el punto (la más pequeña si hay varias)."""
    cand = [t for t in tracks if t.visible and t.bbox[0] <= u <= t.bbox[0] + t.bbox[2]
            and t.bbox[1] <= v <= t.bbox[1] + t.bbox[3]]
    return min(cand, key=lambda t: t.bbox[2] * t.bbox[3]) if cand else None


def select_nearest(tracks: List[Track]) -> Optional[Track]:
    """Track visible más cercano (caja más grande)."""
    vis = [t for t in tracks if t.visible]
    return max(vis, key=lambda t: t.bbox[2] * t.bbox[3]) if vis else None
