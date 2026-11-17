import numpy as np
import yaml

from lite6_handeye import calib_io, cli
from lite6_handeye.transforms import T_from_xyz_rpy

from test_core import _synthetic


def test_samples_roundtrip_and_offline_solve(tmp_path):
    X, _, G, C = _synthetic(np.random.default_rng(7), n=15, noise_t=0.0002, noise_r_deg=0.05)
    T_ro = T_from_xyz_rpy([0.0, 0.06, 0.0], np.deg2rad([-90.0, 0.0, -90.0]))
    frames = {'robot_base_frame': 'link_base', 'robot_effector_frame': 'link_eef',
              'camera_root_frame': 'zed_camera_link', 'optical_frame': 'zed_left_camera_frame_optical'}
    samples = [{'T_base_eef': g, 'T_cam_board': c, 'joints': {'joint1': 0.1}} for g, c in zip(G, C)]
    sf = tmp_path / 'calib_samples.yaml'
    calib_io.save_samples(str(sf), frames, samples, T_ro)
    fr, G2, C2, T_ro2 = calib_io.load_samples(str(sf))
    assert fr == frames and np.allclose(G2[3], G[3]) and np.allclose(T_ro2, T_ro)

    cli.solve_offline([str(sf), '--method', 'refine'])
    res = calib_io.load_result(str(tmp_path / 'calib_refine.yaml'))
    assert res['publish']['parent_frame'] == 'link_eef'
    assert res['publish']['child_frame'] == 'zed_camera_link'
    # publicado = X * inv(T_root_optical)
    T_pub = calib_io.tf_dict_to_T(res['publish'])
    assert np.allclose(T_pub @ T_ro, X, atol=2e-3)
    assert '--child-frame-id zed_camera_link' in res['static_transform_publisher']


def test_generate_board_cli(tmp_path):
    out = tmp_path / 'b'
    cli.generate_board(['--out', str(out), '--png-px-per-mm', '5'])
    svg = (tmp_path / 'b.svg').read_text()
    assert svg.startswith('<svg') and 'width="170.0000mm"' in svg
    assert (tmp_path / 'b.png').stat().st_size > 0


def test_poses_roundtrip(tmp_path):
    p = tmp_path / 'poses.yaml'
    calib_io.save_poses(str(p), ['joint1', 'joint2'], [[0.1, 0.2], [0.3, 0.4]])
    names, qs = calib_io.load_poses(str(p))
    assert names == ['joint1', 'joint2'] and qs[1] == [0.3, 0.4]
    assert yaml.safe_load(p.read_text())['poses'][0] == [0.1, 0.2]


def test_node_params_override_wins_over_node_section(tmp_path):
    from lite6_handeye.launch_utils import node_params
    p = tmp_path / 'p.yaml'
    p.write_text('/**:\n  ros__parameters:\n    board.squares_x: 5\n'
                 'calib:\n  ros__parameters:\n    use_board_pose_base: false\n    n_poses: 20\n')
    out = node_params(str(p), 'calib', {'use_board_pose_base': True})
    assert out == {'board.squares_x': 5, 'use_board_pose_base': True, 'n_poses': 20}


def test_sim_camera_and_compare_gt(tmp_path):
    from lite6_handeye import sim_camera
    X, T_link = sim_camera.true_transforms([0, -0.06, 0.04, 0, 0, 0], [0.008, -0.005, 0.006, 0.03, -0.05, 0.07])
    # óptico = link * T_LINK_OPTICAL ; z óptico == x del link
    assert np.allclose(T_link @ sim_camera.T_LINK_OPTICAL, X)
    assert np.allclose(T_link[:3, 0], X[:3, 2])
    urdf = sim_camera.inject('<robot name="r"><link name="link_eef"/></robot>', sim_camera.urdf_snippet(T_link))
    assert urdf.count('<sensor name="sim_camera" type="rgbd_camera">') == 1 and urdf.endswith('</robot>')
    assert urdf.count('<sensor name="sim_depth" type="rgbd_camera">') == 1 and '<width>640</width>' in urdf
    gt = tmp_path / 'gt.yaml'
    calib_io.save_ground_truth(str(gt), X, T_link, {'robot_effector_frame': 'link_eef'})
    res = {'eef_to_optical': calib_io._tf_dict(X @ T_from_xyz_rpy([0.0005, 0, 0], [0, 0, 0])),
           'publish': calib_io._tf_dict(T_link)}
    e = calib_io.compare(res, calib_io.load_result(str(gt)))
    assert abs(e['eef_to_optical']['trans_mm'] - 0.5) < 1e-6 and e['eef_to_camera_root']['trans_mm'] < 1e-6
