"""Generación de poses de calibración alrededor del tablero.

Cada pose de cámara mira al centro del tablero desde un casquete esférico:
  - inclinación respecto a la normal del tablero en [tilt_min, tilt_max]
  - acimut distribuido (ángulo áureo) dentro de +-azimuth_range_deg alrededor de la
    dirección hacia azimuth_towards (p. ej. la base: el lado opuesto queda fuera de alcance)
  - distancia variable en [dist_min, dist_max]
  - giro alrededor del eje óptico en [-roll_max, roll_max]
La pose del efector se obtiene con una estimación previa X0 = T_eef_cam
(nominal de CAD; solo se usa para planificar, no para el resultado).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import inv_T, make_T

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def look_at(cam_pos, target, up_hint):
    """Rotación de cámara óptica (z adelante, y abajo) mirando a target."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(up_hint, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def board_normal_towards(T_base_board, cam_pos_base):
    """Normal del tablero apuntando hacia la cámara (independiente de convención de OpenCV)."""
    n = T_base_board[:3, 2].copy()
    if np.dot(n, cam_pos_base - T_base_board[:3, 3]) < 0:
        n = -n
    return n


def generate_viewpoints(T_base_board, board_center, normal, X0, n_poses=20,
                        dist_range=(0.25, 0.40), tilt_range_deg=(20.0, 45.0),
                        roll_max_deg=30.0, up_hint=None, seed=0,
                        azimuth_towards=None, azimuth_range_deg=180.0, follow_azimuth=False):
    """Devuelve lista de (T_base_eef, T_base_cam).

    azimuth_towards: punto (base) hacia el que se centran los acimuts, visto desde el
    centro del tablero y proyectado en su plano. None o azimuth_range_deg >= 180 -> 360°.
    follow_azimuth: el "arriba" de la cámara gira con el acimut (orbita como si diera la
    vuelta al tablero) en vez de quedar fijo en el mundo.
    """
    rng = np.random.default_rng(seed)
    target = (T_base_board @ np.append(board_center, 1.0))[:3]
    n = normal / np.linalg.norm(normal)
    u = None
    if azimuth_towards is not None:
        a = np.asarray(azimuth_towards, float) - target
        a = a - (a @ n) * n
        if np.linalg.norm(a) > 1e-6:
            u = a / np.linalg.norm(a)
    if u is None:
        u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        azimuth_range_deg = 180.0
    span = np.deg2rad(min(azimuth_range_deg, 180.0))
    v = np.cross(n, u)
    if up_hint is None:
        up_hint = -u  # cualquier dirección del plano; el roll se varía igual

    X0_inv = inv_T(X0)
    out = []
    for k in range(n_poses):
        # inclinación estratificada en [tilt_min, tilt_max], alternando extremos
        frac = (k + 0.5) / n_poses
        tilt = np.deg2rad(tilt_range_deg[0] + (tilt_range_deg[1] - tilt_range_deg[0])
                          * ((frac * 7.0) % 1.0))
        if span >= np.pi:
            az = k * GOLDEN_ANGLE
        else:  # secuencia áurea en [-span, span]
            az = -span + 2.0 * span * ((k * (np.sqrt(5.0) - 1.0) / 2.0 + 0.5 / n_poses) % 1.0)
        dist = rng.uniform(*dist_range)
        d = np.cos(tilt) * n + np.sin(tilt) * (np.cos(az) * u + np.sin(az) * v)
        cam_pos = target + dist * d
        up = Rotation.from_rotvec(n * az).apply(up_hint) if follow_azimuth else up_hint
        R = look_at(cam_pos, target, up)
        roll = np.deg2rad(rng.uniform(-roll_max_deg, roll_max_deg))
        R = R @ Rotation.from_rotvec([0.0, 0.0, roll]).as_matrix()
        T_base_cam = make_T(R, cam_pos)
        out.append((az % (2 * np.pi) if span >= np.pi else az, T_base_cam @ X0_inv, T_base_cam))
    # recorrido continuo alrededor del tablero (menos giro de muñeca entre poses)
    out.sort(key=lambda o: o[0])
    return [(T_eef, T_cam) for _, T_eef, T_cam in out]


def observation_viewpoints(target, X0, dist_range=(0.65, 0.80), elev_range_deg=(40.0, 85.0),
                           azimuth_towards=(0.0, 0.0, 0.0), azimuth_range_deg=90.0,
                           n_dist=3, n_elev=10, n_az=7,
                           rolls_deg=(0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)):
    """Vistas para observar un objeto con un sensor de profundidad de alcance mínimo dist_range[0].

    La cámara mira a target desde una dirección con elevación (sobre el plano xy de la base) en
    elev_range_deg y acimut centrado hacia azimuth_towards (la base del robot: del otro lado no
    alcanza). Devuelve (T_base_eef, T_base_cam) ordenadas por preferencia: más cenital (la cara
    superior se ve menos rasante), más cerca y más centrada; el roll varía al final (alcanzabilidad).
    """
    target = np.asarray(target, float)
    a = np.asarray(azimuth_towards, float)[:2] - target[:2]
    az0 = np.arctan2(a[1], a[0]) if np.linalg.norm(a) > 1e-6 else 0.0
    span = np.deg2rad(azimuth_range_deg)
    azs = sorted(np.linspace(-span, span, n_az), key=abs)
    X0_inv = inv_T(X0)
    out = []
    for elev in np.deg2rad(np.linspace(elev_range_deg[1], elev_range_deg[0], n_elev)):
        for dist in np.linspace(*dist_range, n_dist):
            for daz in azs:
                az = az0 + daz
                d = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
                cam_pos = target + dist * d
                # y óptico hacia abajo: "arriba" de la imagen = lejos de la base
                R = look_at(cam_pos, target, np.array([0.0, 0.0, -1.0]))
                for roll in rolls_deg:
                    Rr = R @ Rotation.from_rotvec([0.0, 0.0, np.deg2rad(roll)]).as_matrix()
                    T_base_cam = make_T(Rr, cam_pos)
                    out.append((T_base_cam @ X0_inv, T_base_cam))
    return out
