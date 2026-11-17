"""Solver Eye-in-Hand AX = XB.

Notación (por muestra i):
  G_i = T_base_eef   (cinemática directa, TF2: link_base -> link_eef)
  C_i = T_cam_board  (detección ChArUco: frame óptico -> tablero)
  X   = T_eef_cam    (incógnita: link_eef -> frame óptico)

El tablero es fijo en la base:  G_i X C_i = T_base_board  para todo i.
Para cada par (i, j):  A = G_j^-1 G_i ,  B = C_j C_i^-1  =>  A X = X B.

Métodos:
  park    Park-Martin (cerrado): rotación por Procrustes sobre logaritmos de
          rotación, traslación por mínimos cuadrados.
  refine  Park-Martin + refinamiento no lineal conjunto de X y T_base_board
          (minimiza error de posición de las esquinas del tablero en la base).
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .transforms import inv_T, make_T, mean_T, rotation_angle


@dataclass
class EvalReport:
    n_samples: int
    trans_rms_mm: float
    trans_max_mm: float
    rot_rms_deg: float
    rot_max_deg: float
    per_sample_trans_mm: list
    per_sample_rot_deg: list

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def _motion_pairs(G, C, min_angle_rad):
    A, B = [], []
    for i, j in combinations(range(len(G)), 2):
        Aij = inv_T(G[j]) @ G[i]
        Bij = C[j] @ inv_T(C[i])
        if rotation_angle(Aij[:3, :3]) < min_angle_rad:
            continue
        A.append(Aij)
        B.append(Bij)
    return A, B


def solve_park_martin(G, C, min_pair_angle_deg=5.0):
    """Devuelve X = T_eef_cam (4x4)."""
    if len(G) != len(C):
        raise ValueError('G y C deben tener la misma longitud')
    if len(G) < 3:
        raise ValueError('Se necesitan al menos 3 muestras')
    A, B = _motion_pairs(G, C, np.deg2rad(min_pair_angle_deg))
    if len(A) < 2:
        raise ValueError(
            'Muy pocos pares con rotación suficiente; aumente la variación angular de las poses')

    alpha = np.array([Rotation.from_matrix(a[:3, :3]).as_rotvec() for a in A])
    beta = np.array([Rotation.from_matrix(b[:3, :3]).as_rotvec() for b in B])
    # alpha_k = R_X beta_k  ->  Procrustes (Kabsch)
    H = beta.T @ alpha
    if np.linalg.matrix_rank(H, tol=1e-9) < 2:
        raise ValueError('Ejes de rotación degenerados; varíe roll/pitch/yaw entre poses')
    U, _, Vt = np.linalg.svd(H)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R_X = Vt.T @ S @ U.T

    # (R_A - I) t_X = R_X t_B - t_A
    M = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    v = np.concatenate([R_X @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    t_X, *_ = np.linalg.lstsq(M, v, rcond=None)
    return make_T(R_X, t_X)


def board_in_base(G, C, X):
    return [g @ X @ c for g, c in zip(G, C)]


def _pack(T):
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def _unpack(p):
    return make_T(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6])


def refine(G, C, X0, board_points=None, loss='huber', f_scale=0.002):
    """Refinamiento no lineal de X y T_base_board.

    board_points: puntos (N,3) en el frame del tablero usados como residuo
    (por defecto un cuadrado de 10 cm). Residuo en metros.
    """
    if board_points is None:
        board_points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0]], float)
    P = np.hstack([board_points, np.ones((len(board_points), 1))]).T  # 4xN
    W0 = mean_T(board_in_base(G, C, X0))

    def residuals(p):
        X = _unpack(p[:6])
        W = _unpack(p[6:])
        target = (W @ P)[:3]
        res = [((g @ X @ c @ P)[:3] - target).ravel() for g, c in zip(G, C)]
        return np.concatenate(res)

    sol = least_squares(residuals, np.concatenate([_pack(X0), _pack(W0)]),
                        loss=loss, f_scale=f_scale, x_scale='jac')
    return _unpack(sol.x[:6]), _unpack(sol.x[6:])


def evaluate(G, C, X):
    """Consistencia: dispersión de T_base_board estimado por cada muestra."""
    W = board_in_base(G, C, X)
    Wm = mean_T(W)
    dt = np.array([np.linalg.norm(w[:3, 3] - Wm[:3, 3]) for w in W]) * 1000.0
    dr = np.array([np.rad2deg(rotation_angle(Wm[:3, :3].T @ w[:3, :3])) for w in W])
    return EvalReport(
        n_samples=len(W),
        trans_rms_mm=float(np.sqrt(np.mean(dt ** 2))),
        trans_max_mm=float(dt.max()),
        rot_rms_deg=float(np.sqrt(np.mean(dr ** 2))),
        rot_max_deg=float(dr.max()),
        per_sample_trans_mm=[float(x) for x in dt],
        per_sample_rot_deg=[float(x) for x in dr],
    )


def solve(G, C, method='refine', board_points=None, outlier_sigma=3.0, min_pair_angle_deg=5.0):
    """Resuelve, descarta outliers una vez (MAD) y re-resuelve.

    Devuelve (X, report, indices_usados).
    """
    G = [np.asarray(g, float) for g in G]
    C = [np.asarray(c, float) for c in C]
    idx = list(range(len(G)))

    def _run(ids):
        g = [G[i] for i in ids]
        c = [C[i] for i in ids]
        X = solve_park_martin(g, c, min_pair_angle_deg)
        if method == 'refine':
            X, _ = refine(g, c, X, board_points)
        elif method != 'park':
            raise ValueError(f'método desconocido: {method}')
        return X, evaluate(g, c, X)

    X, rep = _run(idx)
    if outlier_sigma and len(idx) > 6:
        e = np.array(rep.per_sample_trans_mm)
        med = np.median(e)
        mad = np.median(np.abs(e - med)) * 1.4826 + 1e-6
        keep = [i for i, ei in zip(idx, e) if ei <= med + outlier_sigma * mad]
        if len(idx) > len(keep) >= 6:
            idx = keep
            X, rep = _run(idx)
    return X, rep, idx
