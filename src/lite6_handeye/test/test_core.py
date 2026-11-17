import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lite6_handeye import charuco, handeye_solver, person_tracker, pose_sampler
from lite6_handeye.transforms import T_from_xyz_rpy, inv_T, make_T, rotation_angle


def _random_T(rng, t_scale=0.3, rot_deg=40):
    R = Rotation.from_rotvec(rng.normal(size=3) * np.deg2rad(rot_deg) / np.sqrt(3)).as_matrix()
    return make_T(R, rng.normal(size=3) * t_scale)


def _synthetic(rng, n=20, noise_t=0.0, noise_r_deg=0.0):
    X = T_from_xyz_rpy([0.012, -0.065, 0.041], np.deg2rad([2.0, -1.5, 90.0]))
    W = T_from_xyz_rpy([0.30, 0.05, 0.0], np.deg2rad([180.0, 0.0, 15.0]))
    G, C = [], []
    for _ in range(n):
        g = make_T(Rotation.from_rotvec(rng.normal(size=3) * 0.5).as_matrix()
                   @ Rotation.from_euler('x', 180, degrees=True).as_matrix(),
                   [0.3, 0.0, 0.3] + rng.normal(size=3) * 0.05)
        c = inv_T(g @ X) @ W
        if noise_t or noise_r_deg:
            dR = Rotation.from_rotvec(rng.normal(size=3) * np.deg2rad(noise_r_deg)).as_matrix()
            c = make_T(dR @ c[:3, :3], c[:3, 3] + rng.normal(size=3) * noise_t)
        G.append(g)
        C.append(c)
    return X, W, G, C


def test_park_martin_exact():
    X, _, G, C = _synthetic(np.random.default_rng(1))
    Xe = handeye_solver.solve_park_martin(G, C)
    assert np.allclose(Xe, X, atol=1e-9)


@pytest.mark.parametrize('method', ['park', 'refine'])
def test_solver_with_noise(method):
    X, _, G, C = _synthetic(np.random.default_rng(2), n=25, noise_t=0.0005, noise_r_deg=0.1)
    Xe, rep, used = handeye_solver.solve(G, C, method=method)
    assert np.linalg.norm(Xe[:3, 3] - X[:3, 3]) < 0.002
    assert np.rad2deg(rotation_angle(Xe[:3, :3].T @ X[:3, :3])) < 0.3
    assert rep.trans_rms_mm < 3.0


def test_solver_rejects_outlier():
    X, _, G, C = _synthetic(np.random.default_rng(3), n=20, noise_t=0.0003, noise_r_deg=0.05)
    C[5] = C[5] @ T_from_xyz_rpy([0.03, 0.0, 0.0], [0.0, 0.0, 0.2])
    Xe, _, used = handeye_solver.solve(G, C, method='refine')
    assert 5 not in used
    assert np.linalg.norm(Xe[:3, 3] - X[:3, 3]) < 0.002


def test_solver_degenerate_rotation_raises():
    rng = np.random.default_rng(4)
    X, W, _, _ = _synthetic(rng, n=1)
    G, C = [], []
    for k in range(6):  # solo traslaciones: no observable
        g = make_T(np.eye(3), [0.3 + 0.01 * k, 0, 0.3])
        G.append(g)
        C.append(inv_T(g @ X) @ W)
    with pytest.raises(ValueError):
        handeye_solver.solve_park_martin(G, C)


CFG = charuco.BoardConfig(5, 7, 0.030, 0.022, 'DICT_4X4_50')


def test_board_config_validation():
    with pytest.raises(ValueError):
        charuco.BoardConfig(5, 6, 0.03, 0.022)
    with pytest.raises(ValueError):
        charuco.BoardConfig(5, 7, 0.03, 0.031)
    with pytest.raises(ValueError):
        charuco.BoardConfig(11, 11, 0.03, 0.022, 'DICT_4X4_50')


def test_layout_matches_opencv_generate_image():
    board, _ = charuco.make_board(CFG)
    px = 10  # px/mm -> cuadrado 300 px, marcador 220 px, bit = 220/6 px
    ref = charuco.draw_board(board, (CFG.squares_x * 300, CFG.squares_y * 300))
    ours = charuco.render_raster(CFG, px_per_mm=px, margin=0.0)
    assert ours.shape == ref.shape
    # Diferencias solo por redondeo de bordes de bit (<0.5 %)
    assert np.mean(ours != ref) < 0.005


def _project_board(img_board, px_per_mm, margin, K, T_cam_board, size):
    """Proyecta la imagen plana del tablero a una cámara pinhole (homografía)."""
    s = 1.0 / (px_per_mm * 1000.0)
    # píxel del tablero (u, v) -> punto del tablero (m) en convención de OpenCV del objeto
    A = np.array([[s, 0, -margin], [0, s, -margin], [0, 0, 1]])
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ A
    return cv2.warpPerspective(img_board, H, size, flags=cv2.INTER_AREA, borderValue=255)


def test_detection_recovers_pose():
    est = charuco.CharucoPoseEstimator(CFG)
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    D = np.zeros(5)
    T = T_from_xyz_rpy([-0.07, -0.10, 0.35], np.deg2rad([20.0, -15.0, 5.0]))
    img = _project_board(charuco.render_raster(CFG, 10, 0.01), 10, 0.01, K, T, (1280, 720))
    det = est.estimate(img, K, D)
    assert det is not None
    assert det.n_corners == (CFG.squares_x - 1) * (CFG.squares_y - 1)
    assert det.reproj_rmse_px < 0.5
    assert np.linalg.norm(det.T_cam_board[:3, 3] - T[:3, 3]) < 0.002
    assert np.rad2deg(rotation_angle(det.T_cam_board[:3, :3].T @ T[:3, :3])) < 0.5


def test_svg_dimensions_and_rects():
    svg = charuco.generate_svg(CFG, margin=0.01)
    root = ET.fromstring(svg)
    assert root.get('width') == '170.0000mm'
    assert root.get('height') == '230.0000mm'
    rects = root.findall('.//{http://www.w3.org/2000/svg}rect')
    black_squares = [r for r in rects if r.get('width') == '30.0000' and r.get('height') == '30.0000']
    assert len(black_squares) == 18  # 35 cuadrados, 18 negros, 17 marcadores
    assert re.search(r'DICT_4X4_50', svg)


def test_viewpoints_look_at_board():
    W = T_from_xyz_rpy([0.30, 0.0, 0.0], [np.pi, 0.0, 0.0])  # tablero con z hacia abajo (OpenCV)
    X0 = T_from_xyz_rpy([0.0, -0.06, 0.04], [0, 0, 0])
    normal = pose_sampler.board_normal_towards(W, np.array([0.3, 0.0, 0.5]))
    assert np.allclose(normal, [0, 0, 1], atol=1e-9)
    vps = pose_sampler.generate_viewpoints(W, CFG.center, normal, X0, n_poses=20)
    target = (W @ np.append(CFG.center, 1))[:3]
    tilts = []
    for T_eef, T_cam in vps:
        assert np.allclose(T_eef @ X0, T_cam)
        z = T_cam[:3, 2]
        to_target = target - T_cam[:3, 3]
        assert np.allclose(z, to_target / np.linalg.norm(to_target), atol=1e-9)
        tilts.append(np.rad2deg(np.arccos(-z @ normal)))
        assert 0.25 - 1e-9 <= np.linalg.norm(to_target) <= 0.40 + 1e-9
    assert 19.9 <= min(tilts) and max(tilts) <= 45.1
    assert max(tilts) - min(tilts) > 15


def test_viewpoints_azimuth_towards_base():
    W = T_from_xyz_rpy([0.30, 0.0, 0.0], [np.pi, 0.0, 0.0])
    X0 = T_from_xyz_rpy([0.0, -0.06, 0.04], [0, 0, 0])
    vps = pose_sampler.generate_viewpoints(W, CFG.center, np.array([0, 0, 1.0]), X0, n_poses=20,
                                           azimuth_towards=np.zeros(3), azimuth_range_deg=90.0)
    target = (W @ np.append(CFG.center, 1))[:3]
    to_base = -target[:2] / np.linalg.norm(target[:2])
    for _, T_cam in vps:
        d = T_cam[:2, 3] - target[:2]
        cos = d @ to_base / np.linalg.norm(d)
        assert cos >= np.cos(np.deg2rad(90.0)) - 1e-9   # cámara del lado de la base


def test_viewpoints_follow_azimuth_rolls_with_orbit():
    W = T_from_xyz_rpy([0.225, 0.105, 0.0], [np.pi, 0.0, 0.0])
    X0 = np.eye(4)
    up = np.array([0, 1.0, 0])
    kw = dict(n_poses=10, roll_max_deg=0.0, up_hint=up, azimuth_towards=np.zeros(3), azimuth_range_deg=90.0)
    fixed = pose_sampler.generate_viewpoints(W, CFG.center, np.array([0, 0, 1.0]), X0, **kw)
    follow = pose_sampler.generate_viewpoints(W, CFG.center, np.array([0, 0, 1.0]), X0, follow_azimuth=True, **kw)
    yaw = lambda T: np.arctan2(T[1, 0], T[0, 0])  # rumbo del eje x de la cámara en el plano  # noqa: E731
    spread = lambda vps: np.ptp(np.unwrap([yaw(Tc) for _, Tc in vps]))  # noqa: E731
    assert spread(follow) > spread(fixed) + np.deg2rad(60)  # el rumbo acompaña la órbita
    for (_, a), (_, b) in zip(fixed, follow):
        assert np.allclose(a[:3, 3], b[:3, 3]) and np.allclose(a[:3, 2], b[:3, 2])  # misma posición y eje óptico


def test_observation_viewpoints_respect_depth_range():
    X0 = T_from_xyz_rpy([0.0125, -0.070, 0.042], [0, 0, np.pi])
    target = np.array([0.30, -0.10, 0.06])
    vps = pose_sampler.observation_viewpoints(target, X0, dist_range=(0.65, 0.80))
    assert len(vps) > 100
    for T_eef, T_cam in vps:
        np.testing.assert_allclose(T_eef @ X0, T_cam, atol=1e-9)
        ray = target - T_cam[:3, 3]
        dist = np.linalg.norm(ray)
        assert 0.65 - 1e-9 <= dist <= 0.80 + 1e-9
        np.testing.assert_allclose(T_cam[:3, 2], ray / dist, atol=1e-9)   # mira al objetivo
        assert T_cam[2, 3] > target[2]                                     # desde arriba
    # la primera es la más cenital
    first = vps[0][1]
    assert -first[2, 2] > np.sin(np.deg2rad(84.0))


def test_person_detector_humanoid():
    img = np.full((720, 1280, 3), 195, dtype=np.uint8)
    # Dibuja humanoide desplazado a la derecha (u=820, cabeza en v=240, torso debajo)
    cv2.circle(img, (820, 240), 36, (125, 160, 215), -1)             # cabeza tono piel (BGR)
    cv2.rectangle(img, (740, 285), (900, 490), (200, 80, 25), -1)    # torso azul (BGR)
    K = np.array([[980.0, 0, 640.0], [0, 980.0, 360.0], [0, 0, 1.0]])
    det = person_tracker.PersonDetector().detect(img, K)
    assert det is not None
    assert det.method == 'humanoid'
    assert abs(det.head_uv[0] - 820.0) < 10.0
    assert abs(det.track_uv[0] - 820.0) < 12.0
    assert 240.0 <= det.track_uv[1] <= 300.0
    assert 1.2 <= det.est_distance <= 2.2


def test_person_tracker_6dof_kinematics_and_ray():
    X = T_from_xyz_rpy([0.0125, -0.070, 0.042], [0.0, 0.0, np.pi])
    K = np.array([[980.0, 0, 640.0], [0, 980.0, 360.0], [0, 0, 1.0]])
    q0 = np.array(person_tracker.WATCH_JOINTS, dtype=float)
    T_eef0 = person_tracker.lite6_fk(q0)
    # En WATCH_JOINTS el eje óptico (+z de cámara) apunta hacia +x de link_base
    T_cam0 = T_eef0 @ X
    np.testing.assert_allclose(T_cam0[:3, 2], [1.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(T_cam0[:3, 1], [0.0, 0.0, -1.0], atol=1e-4)

    # Punto en el centro óptico a 1.2 m delante de la cámara
    p_base = person_tracker.ray_to_target_3d(640.0, 360.0, K, T_eef0, X, 1.2)
    np.testing.assert_allclose(p_base, T_cam0[:3, 3] + np.array([1.2, 0.0, 0.0]), atol=1e-6)

    # Seguimiento 6-DOF cuando la persona se desplaza en y y en z
    for target in ([1.35, -0.45, 0.42], [1.35, 0.0, 0.38], [1.35, 0.45, 0.22]):
        tgt = np.array(target, dtype=float)
        T_eef_des, T_cam_des = person_tracker.compute_tracking_pose_6dof(tgt, X)
        np.testing.assert_allclose(T_eef_des @ X, T_cam_des, atol=1e-9)
        # El eje óptico (+z) apunta exactamente a la persona
        ray = tgt - T_cam_des[:3, 3]
        np.testing.assert_allclose(T_cam_des[:3, 2], ray / np.linalg.norm(ray), atol=1e-6)
        q_sol, ok = person_tracker.lite6_ik_6dof(T_eef_des, q0)
        assert ok
        np.testing.assert_allclose(person_tracker.lite6_fk(q_sol), T_eef_des, atol=1e-3)


def test_usd_person_import(tmp_path):
    import os
    from lite6_handeye import usd_person

    usd_file = str(tmp_path / 'sample_person.usda')
    model_dir = str(tmp_path / 'person_target')
    os.makedirs(model_dir, exist_ok=True)

    usd_person.create_sample_realistic_usd(usd_file)
    assert os.path.exists(usd_file)

    info = usd_person.import_usd_to_person_target(
        usd_path=usd_file,
        model_dir=model_dir,
        target_height=1.72,
        ground_z=-0.75,
    )
    assert info['submeshes'] == 18
    assert info['triangles'] > 2000
    assert os.path.exists(info['sdf_path'])
    assert os.path.exists(os.path.join(model_dir, 'meshes', 'usd_person.mtl'))
    with open(info['sdf_path']) as f:
        sdf_txt = f.read()
    for link in usd_person.LINKS:
        assert f'<link name="{link}">' in sdf_txt
        assert f'model://person_target/meshes/usd_{link}_' in sdf_txt
    for j in usd_person.JOINTS:
        assert f'<joint name="{j}"' in sdf_txt


def test_kalman_smooths_noisy_static_target():
    rng = np.random.default_rng(0)
    kf = person_tracker.TargetKalman(accel_std=0.3, meas_lat_std=0.03, meas_depth_std=0.15)
    truth = np.array([1.2, 0.1, 0.4])
    ray = np.array([1.0, 0.0, 0.0])
    raw_err, kf_err = [], []
    for k in range(150):
        noise = rng.normal(0, 0.03, 3) + ray * rng.normal(0, 0.15)
        kf.update(truth + noise, k / 15.0, ray_dir=ray)
        if k >= 50:
            raw_err.append(np.linalg.norm(noise))
            kf_err.append(np.linalg.norm(kf.position - truth))
    assert np.mean(kf_err) < 0.4 * np.mean(raw_err)


def test_kalman_tracks_constant_velocity():
    kf = person_tracker.TargetKalman(accel_std=0.5)
    v = np.array([0.0, 0.3, 0.0])
    for k in range(60):
        t = k / 15.0
        kf.update(np.array([1.0, 0.0, 0.4]) + v * t, t)
    assert np.allclose(kf.position, np.array([1.0, 0.0, 0.4]) + v * 59 / 15.0, atol=0.02)
    assert np.allclose(kf.x[3:], v, atol=0.05)


def test_kalman_gates_outlier_then_resets():
    kf = person_tracker.TargetKalman(gate_sigma=3.5, max_rejects=3)
    for k in range(30):
        kf.update(np.array([1.0, 0.0, 0.4]), k / 15.0)
    p0 = kf.position
    assert not kf.update(np.array([1.0, 1.5, 0.4]), 2.0)      # salto de 1.5 m: descartado
    assert np.allclose(kf.position[:2], p0[:2], atol=1e-6)
    assert not kf.update(np.array([1.0, 1.5, 0.4]), 2.07)
    assert kf.update(np.array([1.0, 1.5, 0.4]), 2.13)        # 3er descarte seguido: reinicia
    assert np.allclose(kf.position, [1.0, 1.5, 0.4])


@pytest.mark.parametrize('use_ruckig', [True, False])
def test_joint_smoother_continuous_and_bounded(use_ruckig):
    from lite6_handeye.motion_smoother import JointSmoother
    dt = 0.02
    sm = JointSmoother(6, dt, max_vel=0.5, max_acc=1.5, max_jerk=8.0, use_ruckig=use_ruckig)
    sm.reset(np.zeros(6))
    goals = [np.full(6, 0.3), np.array([0.6, -0.2, 0.1, 0.0, 0.3, -0.4])]   # cambio de meta en marcha
    vs = []
    for k in range(400):
        if k == 0:
            sm.set_goal(goals[0])
        if k == 25:
            sm.set_goal(goals[1])
        q, v, _, _ = sm.step(lookahead=0.1)
        vs.append(v)
    vs = np.array(vs)
    acc = np.diff(vs, axis=0) / dt
    assert np.max(np.abs(vs)) <= 0.5 + 1e-6
    # velocidad continua al cambiar la meta (el respaldo sin ruckig puede frenar un paso de más al llegar)
    assert np.max(np.abs(acc)) <= (1.5 + 1e-3 if sm.backend == 'ruckig' else 3.0)
    assert np.allclose(q, goals[1], atol=1e-3) and sm.settled(1e-3)


def _pdet(x, y, w=80, h=200, score=0.9):
    return person_tracker.PersonDetection(bbox=(x, y, w, h), track_uv=(x + w / 2, y + 20),
                                          head_uv=(x + w / 2, y + 20), est_distance=2.0,
                                          score=score, method='yolo')


def test_bytetrack_keeps_ids_when_people_cross():
    from lite6_handeye.byte_tracker import ByteTracker
    bt = ByteTracker()
    ids_a, ids_b = set(), set()
    for k in range(40):                       # A va a la derecha, B a la izquierda; se cruzan en k=20
        a, b = _pdet(100 + 10 * k, 200), _pdet(500 - 10 * k, 220)
        tracks = bt.update([a, b] if k % 7 else [a, b][::-1])
        for t in tracks:
            if t.visible and t.det is a:
                ids_a.add(t.id)
            if t.visible and t.det is b:
                ids_b.add(t.id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_bytetrack_recovers_low_score_and_short_loss():
    from lite6_handeye.byte_tracker import ByteTracker
    bt = ByteTracker(track_buffer=30)
    for k in range(10):
        bt.update([_pdet(100 + 5 * k, 200)])
    tid = bt.tracks[0].id
    bt.update([_pdet(150, 200, score=0.25)])   # ocluida: puntaje bajo, mismo ID (2.ª etapa)
    assert bt.tracks[0].visible and bt.tracks[0].id == tid
    for _ in range(10):                        # desaparece 10 cuadros
        bt.update([])
    t = [t for t in bt.update([_pdet(205, 200)]) if t.visible]
    assert len(t) == 1 and t[0].id == tid      # reaparece cerca: conserva el ID


def _person_img(people, size=(720, 1280)):
    """Imagen gris con personas sintéticas: (x, y, w, h, color_torso, color_piernas) en BGR."""
    img = np.full(size + (3,), 90, np.uint8)
    for x, y, w, h, top, bottom in people:
        img[y:y + h // 2, x:x + w] = top
        img[y + h // 2:y + h, x:x + w] = bottom
    return img


def _pdet_feat(img, x, y, w=80, h=200, score=0.9):
    from lite6_handeye.byte_tracker import appearance_feature
    d = _pdet(x, y, w, h, score)
    d.feat = appearance_feature(img, d.bbox)
    return d


def test_reid_gallery_restores_id_after_leaving_frame():
    from lite6_handeye.byte_tracker import ByteTracker
    bt = ByteTracker(track_buffer=10, reid_buffer=300, reid_probation=5)
    red, blue, green, yellow = (40, 40, 200), (180, 60, 30), (40, 180, 40), (30, 200, 220)
    for k in range(20):                          # A (camisa roja, pantalón azul) a la vista
        img = _person_img([(200 + 2 * k, 200, 80, 200, red, blue)])
        bt.update([_pdet_feat(img, 200 + 2 * k, 200)])
    tid_a = bt.tracks[0].id
    for _ in range(30):                          # sale del cuadro más que track_buffer
        bt.update([])
    assert bt.get(tid_a) is None and tid_a in bt.gallery and bt.known(tid_a)
    # B (otra ropa) entra: ID nuevo
    img = _person_img([(900, 200, 80, 200, green, yellow)])
    tb = [t for t in bt.update([_pdet_feat(img, 900, 200)]) if t.visible]
    assert tb[0].id != tid_a
    # A vuelve por el otro lado (lejos de donde se perdió): recupera su ID
    for k in range(3):
        img = _person_img([(900, 200, 80, 200, green, yellow), (100, 210, 80, 200, red, blue)])
        tracks = bt.update([_pdet_feat(img, 900, 200), _pdet_feat(img, 100, 210)])
    ids = {t.id for t in tracks if t.visible}
    assert tid_a in ids and tb[0].id in ids and len(ids) == 2
    assert tid_a not in bt.gallery


def test_reid_lost_track_reappears_far_away():
    from lite6_handeye.byte_tracker import ByteTracker
    bt = ByteTracker(track_buffer=60, reid_probation=5)
    red, blue = (40, 40, 200), (180, 60, 30)
    for k in range(20):
        img = _person_img([(200, 200, 80, 200, red, blue)])
        bt.update([_pdet_feat(img, 200, 200)])
    tid = bt.tracks[0].id
    for _ in range(15):                          # tapada 0.5 s
        bt.update([])
    img = _person_img([(1000, 220, 80, 200, red, blue)])
    for _ in range(3):                           # reaparece sin solapar: IoU no la recupera;
        tracks = bt.update([_pdet_feat(img, 1000, 220)])   # se confirma tras reid_confirm cuadros
    assert [t.id for t in tracks] == [tid]


def test_tracking_pose_full_azimuth_and_unwrap():
    X = T_from_xyz_rpy([0.0125, -0.07, 0.042], [0, 0, np.pi])
    behind = np.array([-1.5, 0.05, 0.4])                       # a la espalda del robot
    _, Tc = person_tracker.compute_tracking_pose_6dof(behind, X, max_azimuth_deg=170)
    assert np.degrees(abs(np.arctan2(Tc[1, 3], Tc[0, 3]))) > 165   # la base gira hacia atrás
    # cruzando ±180° desde +170°: no da la vuelta larga hacia -170°
    p = np.array([1.5 * np.cos(np.radians(-175)), 1.5 * np.sin(np.radians(-175)), 0.4])
    _, Tc = person_tracker.compute_tracking_pose_6dof(p, X, max_azimuth_deg=170, az_ref=np.radians(170))
    assert np.degrees(np.arctan2(Tc[1, 3], Tc[0, 3])) > 165


def test_bytetrack_camera_motion_compensation():
    from lite6_handeye.byte_tracker import ByteTracker
    from scipy.spatial.transform import Rotation as R
    K = np.array([[967.8, 0, 632.4], [0, 966.4, 364.6], [0, 0, 1.0]])
    for cmc in (False, True):
        bt = ByteTracker()
        for _ in range(5):
            bt.update([_pdet(600, 250)])
        tid = bt.tracks[0].id
        # la cámara gira 8° en yaw: la persona (quieta) se desplaza ~135 px en la imagen
        R_rel = R.from_euler('y', -8, degrees=True).as_matrix()
        H = K @ R_rel @ np.linalg.inv(K)
        c = H @ np.array([640.0, 350.0, 1.0])
        moved = _pdet(int(c[0] / c[2] - 40), 250)
        if cmc:
            bt.apply_homography(H)
        vis = [t for t in bt.update([moved]) if t.visible]
        assert (vis[0].id == tid) == cmc            # sin compensación: ID nuevo; con: el mismo
