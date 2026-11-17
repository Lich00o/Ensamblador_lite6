"""Tablero ChArUco: configuración, generación vectorial (SVG para láser) y
estimación de pose.

Compatible con OpenCV 4.6 (python3-opencv de Ubuntu 24.04, API aruco antigua)
y OpenCV >= 4.7 (CharucoDetector).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .transforms import T_from_rvec_tvec

_NEW_API = hasattr(cv2.aruco, 'CharucoDetector')


@dataclass
class BoardConfig:
    squares_x: int = 5
    squares_y: int = 7
    square_length: float = 0.030   # m, MEDIDO post-grabado
    marker_length: float = 0.022   # m, MEDIDO post-grabado
    dictionary: str = 'DICT_4X4_50'

    def __post_init__(self):
        if self.squares_y % 2 == 0:
            # OpenCV < 4.7 y >= 4.7 dibujan distinto los tableros con nº de filas par
            # ("legacy pattern"). Con filas impares el patrón es idéntico en ambas.
            raise ValueError('squares_y debe ser impar (compatibilidad OpenCV 4.6 / >= 4.7)')
        if not 0 < self.marker_length < self.square_length:
            raise ValueError('marker_length debe ser > 0 y < square_length')
        n_markers = (self.squares_x * self.squares_y) // 2
        dict_size = int(self.dictionary.rsplit('_', 1)[-1])
        if n_markers > dict_size:
            raise ValueError(f'{self.dictionary} tiene {dict_size} marcadores; el tablero necesita {n_markers}')

    @classmethod
    def from_params(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @property
    def center(self):
        """Centro del tablero en su propio frame (m)."""
        return np.array([self.squares_x * self.square_length / 2.0,
                         self.squares_y * self.square_length / 2.0, 0.0])

    @property
    def size(self):
        return self.squares_x * self.square_length, self.squares_y * self.square_length


def get_dictionary(name):
    d = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        return cv2.aruco.getPredefinedDictionary(d)
    return cv2.aruco.Dictionary_get(d)


def make_board(cfg):
    dictionary = get_dictionary(cfg.dictionary)
    if _NEW_API:
        board = cv2.aruco.CharucoBoard((cfg.squares_x, cfg.squares_y),
                                       cfg.square_length, cfg.marker_length, dictionary)
    else:
        board = cv2.aruco.CharucoBoard_create(cfg.squares_x, cfg.squares_y,
                                              cfg.square_length, cfg.marker_length, dictionary)
    return board, dictionary


def draw_board(board, size_px, margin_px=0, border_bits=1):
    """Imagen del tablero dibujada por OpenCV (referencia)."""
    if hasattr(board, 'generateImage'):
        return board.generateImage(size_px, marginSize=margin_px, borderBits=border_bits)
    return board.draw(size_px, marginSize=margin_px, borderBits=border_bits)


def board_chessboard_corners(board):
    if hasattr(board, 'getChessboardCorners'):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    return np.asarray(board.chessboardCorners, dtype=np.float64).reshape(-1, 3)


def _marker_bits(dictionary, marker_id, marker_bits):
    """Matriz (n+2)x(n+2) de bits (1 = negro) incluyendo borde negro de 1 bit."""
    n = marker_bits + 2
    if hasattr(cv2.aruco, 'generateImageMarker'):
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, n, borderBits=1)
    else:
        img = cv2.aruco.drawMarker(dictionary, marker_id, n, borderBits=1)
    return (img < 128).astype(np.uint8)


def _dictionary_marker_size(dictionary):
    return int(dictionary.markerSize)


def board_layout(cfg):
    """Lista de cuadrados negros (col, fila) y marcadores (col, fila, id).

    Fila 0 = fila superior del dibujo. Mismo esquema que CharucoBoard.generateImage
    (cuadrado (0,0) negro, marcadores en los blancos, ids en orden fila-mayor).
    """
    black, markers = [], []
    mid = 0
    for row in range(cfg.squares_y):
        for col in range(cfg.squares_x):
            if (row + col) % 2 == 0:
                black.append((col, row))
            else:
                markers.append((col, row, mid))
                mid += 1
    return black, markers


def generate_svg(cfg, margin=0.010, label=True):
    """SVG en milímetros reales (1 unidad = 1 mm). Negro = zona a grabar.

    margin: margen blanco alrededor del tablero (m).
    """
    _, dictionary = make_board(cfg)
    nbits = _dictionary_marker_size(dictionary)
    mm = 1000.0
    L = cfg.square_length * mm
    M = cfg.marker_length * mm
    off = (L - M) / 2.0
    cell = M / (nbits + 2)
    W = cfg.squares_x * L + 2 * margin * mm
    H = cfg.squares_y * L + 2 * margin * mm
    m = margin * mm

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.4f}mm" height="{H:.4f}mm" '
           f'viewBox="0 0 {W:.4f} {H:.4f}" shape-rendering="crispEdges">',
           f'<rect x="0" y="0" width="{W:.4f}" height="{H:.4f}" fill="white"/>',
           '<g fill="black" stroke="none">']
    black, markers = board_layout(cfg)
    for col, row in black:
        out.append(f'<rect x="{m + col * L:.4f}" y="{m + row * L:.4f}" width="{L:.4f}" height="{L:.4f}"/>')
    for col, row, mid in markers:
        bits = _marker_bits(dictionary, mid, nbits)
        x0 = m + col * L + off
        y0 = m + row * L + off
        for r in range(bits.shape[0]):
            # Agrupa bits negros contiguos por fila (menos rectángulos, mismo resultado)
            c = 0
            while c < bits.shape[1]:
                if bits[r, c]:
                    c1 = c
                    while c1 < bits.shape[1] and bits[r, c1]:
                        c1 += 1
                    out.append(f'<rect x="{x0 + c * cell:.4f}" y="{y0 + r * cell:.4f}" '
                               f'width="{(c1 - c) * cell:.4f}" height="{cell:.4f}"/>')
                    c = c1
                else:
                    c += 1
    out.append('</g>')
    if label and m >= 4.0:
        out.append(f'<text x="{m:.2f}" y="{H - m / 3:.2f}" font-family="monospace" '
                   f'font-size="{min(3.0, m / 2):.2f}" fill="black">'
                   f'{cfg.squares_x}x{cfg.squares_y} sq={L:.2f}mm mk={M:.2f}mm {cfg.dictionary}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def render_raster(cfg, px_per_mm=10, margin=0.010):
    """Rasteriza el mismo layout del SVG (para PNG / pruebas). uint8, blanco=255."""
    _, dictionary = make_board(cfg)
    nbits = _dictionary_marker_size(dictionary)
    s = px_per_mm * 1000.0
    L = cfg.square_length * s
    M = cfg.marker_length * s
    m = margin * s
    W = int(round(cfg.squares_x * L + 2 * m))
    H = int(round(cfg.squares_y * L + 2 * m))
    img = np.full((H, W), 255, np.uint8)
    black, markers = board_layout(cfg)
    for col, row in black:
        x0, y0 = int(round(m + col * L)), int(round(m + row * L))
        x1, y1 = int(round(m + (col + 1) * L)), int(round(m + (row + 1) * L))
        img[y0:y1, x0:x1] = 0
    cell = M / (nbits + 2)
    for col, row, mid in markers:
        bits = _marker_bits(dictionary, mid, nbits)
        x0 = m + col * L + (L - M) / 2
        y0 = m + row * L + (L - M) / 2
        for r in range(bits.shape[0]):
            for c in range(bits.shape[1]):
                if bits[r, c]:
                    img[int(round(y0 + r * cell)):int(round(y0 + (r + 1) * cell)),
                        int(round(x0 + c * cell)):int(round(x0 + (c + 1) * cell))] = 0
    return img


@dataclass
class Detection:
    T_cam_board: np.ndarray
    n_corners: int
    reproj_rmse_px: float
    corners: np.ndarray
    ids: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray


class CharucoPoseEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.board, self.dictionary = make_board(cfg)
        self.obj_corners = board_chessboard_corners(self.board)
        if _NEW_API:
            self._detector = cv2.aruco.CharucoDetector(self.board)
        else:
            self._params = cv2.aruco.DetectorParameters_create()
            self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def detect_corners(self, gray, K=None, D=None):
        if _NEW_API:
            corners, ids, _, _ = self._detector.detectBoard(gray)
        else:
            mc, mi, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self._params)
            if mi is None or len(mi) == 0:
                return None, None
            _, corners, ids = cv2.aruco.interpolateCornersCharuco(
                mc, mi, gray, self.board, cameraMatrix=K, distCoeffs=D)
        if ids is None or len(ids) == 0:
            return None, None
        return np.asarray(corners, np.float64).reshape(-1, 2), np.asarray(ids).ravel()

    def estimate(self, gray, K, D, min_corners=6):
        K = np.asarray(K, np.float64).reshape(3, 3)
        D = np.asarray(D, np.float64).ravel()
        corners, ids = self.detect_corners(gray, K, D)
        if corners is None or len(ids) < min_corners:
            return None
        obj = self.obj_corners[ids]
        # Colinealidad: todas las esquinas en una línea -> pose indeterminada
        if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0), tol=1e-6) < 2:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        if hasattr(cv2, 'solvePnPRefineLM'):
            rvec, tvec = cv2.solvePnPRefineLM(obj, corners, K, D, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
        return Detection(T_from_rvec_tvec(rvec, tvec), len(ids), err, corners, ids, rvec, tvec)

    def draw(self, bgr, det, K, D):
        for p in det.corners:
            cv2.circle(bgr, (int(round(p[0])), int(round(p[1]))), 3, (0, 255, 0), -1)
        cv2.drawFrameAxes(bgr, np.asarray(K, float).reshape(3, 3), np.asarray(D, float).ravel(),
                          det.rvec, det.tvec, self.cfg.square_length * 2)
        cv2.putText(bgr, f'n={det.n_corners} rmse={det.reproj_rmse_px:.2f}px', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return bgr
