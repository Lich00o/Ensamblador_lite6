"""Lectura/escritura de muestras, poses y resultados (YAML)."""

import datetime
import os

import numpy as np
import yaml

from . import handeye_solver
from .transforms import T_to_xyz_quat, T_from_xyz_quat, inv_T, rotation_angle


def _T_to_list(T):
    return [float(x) for x in np.asarray(T).ravel()]


def _T_from_list(v):
    return np.asarray(v, float).reshape(4, 4)


def _tf_dict(T):
    xyz, q = T_to_xyz_quat(T)
    return {'translation': {k: float(v) for k, v in zip('xyz', xyz)},
            'rotation': {k: float(v) for k, v in zip('xyzw', q)}}


def tf_dict_to_T(d):
    t, r = d['translation'], d['rotation']
    return T_from_xyz_quat([t['x'], t['y'], t['z']], [r['x'], r['y'], r['z'], r['w']])


def expand(path):
    return os.path.abspath(os.path.expanduser(path))


def save_samples(path, frames, samples, T_root_optical):
    """samples: lista de dict {T_base_eef, T_cam_board, joints, n_detections}."""
    data = {
        'frames': frames,
        'T_camera_root_optical': _T_to_list(T_root_optical) if T_root_optical is not None else None,
        'samples': [{
            'T_base_eef': _T_to_list(s['T_base_eef']),
            'T_cam_board': _T_to_list(s['T_cam_board']),
            'joints': {k: float(v) for k, v in s.get('joints', {}).items()},
            'n_detections': int(s.get('n_detections', 1)),
        } for s in samples],
    }
    _dump(path, data)


def load_samples(path):
    with open(expand(path)) as f:
        data = yaml.safe_load(f)
    G = [_T_from_list(s['T_base_eef']) for s in data['samples']]
    C = [_T_from_list(s['T_cam_board']) for s in data['samples']]
    T_ro = data.get('T_camera_root_optical')
    return data['frames'], G, C, (_T_from_list(T_ro) if T_ro is not None else None)


def save_poses(path, joint_names, joint_targets):
    _dump(path, {'joint_names': list(joint_names),
                 'poses': [[float(x) for x in q] for q in joint_targets]})


def load_poses(path):
    with open(expand(path)) as f:
        data = yaml.safe_load(f)
    return data['joint_names'], [list(map(float, q)) for q in data['poses']]


def solve_and_save(path, name, frames, G, C, T_root_optical, method='refine', board_points=None):
    """Resuelve y escribe el YAML de resultado. Devuelve (dict, X)."""
    X, rep, used = handeye_solver.solve(G, C, method=method, board_points=board_points)
    eef, root, optical = frames['robot_effector_frame'], frames['camera_root_frame'], frames['optical_frame']
    if T_root_optical is not None:
        T_pub, child = X @ inv_T(T_root_optical), root
    else:
        T_pub, child = X, optical
    xyz, q = T_to_xyz_quat(T_pub)
    cmd = ('ros2 run tf2_ros static_transform_publisher '
           f'--x {xyz[0]:.6f} --y {xyz[1]:.6f} --z {xyz[2]:.6f} '
           f'--qx {q[0]:.6f} --qy {q[1]:.6f} --qz {q[2]:.6f} --qw {q[3]:.6f} '
           f'--frame-id {eef} --child-frame-id {child}')
    result = {
        'name': name,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'calibration_type': 'eye_in_hand',
        'method': method,
        'frames': dict(frames),
        'eef_to_optical': _tf_dict(X),
        'publish': {'parent_frame': eef, 'child_frame': child, **_tf_dict(T_pub)},
        'metrics': {k: v for k, v in rep.as_dict().items() if not k.startswith('per_sample')},
        'per_sample_trans_mm': rep.per_sample_trans_mm,
        'samples_used': [int(i) for i in used],
        'n_samples_total': len(G),
        'static_transform_publisher': cmd,
    }
    _dump(path, result)
    return result, X


def save_ground_truth(path, T_eef_optical, T_eef_root, frames):
    _dump(path, {'frames': dict(frames), 'eef_to_optical': _tf_dict(T_eef_optical),
                 'eef_to_camera_root': _tf_dict(T_eef_root)})


def compare(result, gt):
    """Error del resultado vs ground truth: dict con mm y grados (óptico y raíz)."""
    out = {}
    for key_r, key_g in (('eef_to_optical', 'eef_to_optical'), ('publish', 'eef_to_camera_root')):
        Tr, Tg = tf_dict_to_T(result[key_r]), tf_dict_to_T(gt[key_g])
        out[key_g] = {'trans_mm': float(np.linalg.norm(Tr[:3, 3] - Tg[:3, 3]) * 1000.0),
                      'rot_deg': float(np.rad2deg(rotation_angle(Tg[:3, :3].T @ Tr[:3, :3])))}
    return out


def load_result(path):
    with open(expand(path)) as f:
        return yaml.safe_load(f)


def _dump(path, data):
    path = expand(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f'# lite6_handeye — {datetime.datetime.now().isoformat(timespec="seconds")}\n')
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None, width=120)
