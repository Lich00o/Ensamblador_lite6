"""Gazebo Harmonic: Lite 6 + cámara RGB-D simulada (X verdadero conocido) + tablero.

  ros2 launch lite6_handeye sim_gazebo.launch.py                 # GUI de gz + RViz
  ros2 launch lite6_handeye sim_gazebo.launch.py gui:=false rviz:=false

Después:
  ros2 launch lite6_handeye auto_calibrate.launch.py camera:=sim
  ros2 service call /handeye_auto_calibrator/start std_srvs/srv/Trigger
  ros2 run lite6_handeye compare_gt

X verdadero = x_prior_xyz_rpy (del params_file) perturbado por cam_error. Se escribe en
~/.ros/lite6_handeye/sim_ground_truth.yaml. Al terminar de cargar, el robot va a la pose "ready".
"""

import os

import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from lite6_handeye import calib_io, sim_camera, usd_person
from lite6_handeye.launch_utils import node_params
from lite6_handeye.presets import CAMERAS

READY = [0.0, 0.2, 1.0, 0.0, 0.8, 0.0]


def _setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
    from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file

    share = get_package_share_directory('lite6_handeye')
    meshes = os.path.join(share, 'sim', 'meshes')
    params_file = g('params_file')
    cam = CAMERAS['sim']

    # --- X verdadero y ground truth
    prior = node_params(params_file, 'handeye_auto_calibrator').get(
        'x_prior_xyz_rpy', [0.0, -0.06, 0.04, 0.0, 0.0, 0.0])
    err = [float(v) for v in g('cam_error').split()]
    X_true, T_eef_link = sim_camera.true_transforms(prior, err)
    calib_io.save_ground_truth(g('ground_truth_file'), X_true, T_eef_link, {
        'robot_effector_frame': 'link_eef', 'camera_root_frame': cam['camera_root_frame'],
        'optical_frame': cam['optical_frame']})

    # --- robot_description (xarm) + cámara
    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(get_package_share_directory('xarm_controller'), 'config', 'lite6_controllers.yaml'),
        prefix='', add_gripper=False, add_bio_gripper=False, ros_namespace='',
        # 200 Hz (xarm_gazebo usa 1000): /tf a 1000 Hz satura los listeners de TF en Python
        update_rate=200,
        use_sim_time=True, robot_type='lite')
    moveit_config = MoveItConfigsBuilder(
        context=context, controllers_name='fake_controllers', dof='6', robot_type='lite', prefix='',
        hw_ns='ufactory', limited='true', attach_to='world', attach_xyz='"0 0 0"', attach_rpy='"0 0 0"',
        ros2_control_plugin='gz_ros2_control/GazeboSimSystem', ros2_control_params=ros2_control_params,
        # Soporte de cámara (CAD cuna + tapa, marco = link_eef, sin ventosa)
        # masa: soporte PLA ~105 g + 4 M4 con insertos 14 g
        add_other_geometry='true', geometry_type='mesh', geometry_mass='0.12',
        geometry_mesh_filename='file://' + os.path.join(meshes, 'soporte_camara.stl'),
    ).to_moveit_configs()
    spec = sim_camera.ASTRA_PRO_PLUS if g('camera_model') == 'astra_pro_plus' else {}
    desc = sim_camera.inject(
        moveit_config.robot_description['robot_description'],
        sim_camera.body_snippet('file://' + os.path.join(meshes, 'orbbec_astra_pro_plus.stl'))
        + sim_camera.urdf_snippet(T_eef_link, topic='sim_camera', optical_frame=cam['optical_frame'],
                                  **spec))
    moveit_config.robot_description['robot_description'] = desc
    srdf = moveit_config.robot_description_semantic['robot_description_semantic']
    moveit_config.robot_description_semantic['robot_description_semantic'] = sim_camera.inject(
        srdf, sim_camera.srdf_disable(['link6', 'other_geometry_link']))
    moveit_config_dump = yaml.dump(moveit_config.to_dict())

    # --- Importación automática de persona en formato OpenUSD (.usd/.usda/.usdc/.usdz)
    person_model_dir = os.path.join(share, 'sim', 'models', 'person_target')
    usd_arg = g('person_usd').strip()
    if not usd_arg:
        user_usd_dir = os.path.expanduser('~/.ros/lite6_handeye/person_usd')
        if os.path.isdir(user_usd_dir):
            pref = os.path.join(user_usd_dir, 'male_adult_construction_01.usd')
            if os.path.exists(pref):
                usd_arg = pref
            else:
                found = sorted(
                    os.path.join(user_usd_dir, f) for f in os.listdir(user_usd_dir)
                    if f.lower().endswith(('.usd', '.usda', '.usdc', '.usdz'))
                )
                if found:
                    usd_arg = found[0]
        if not usd_arg:
            default_usd = os.path.join(person_model_dir, 'usd', 'realistic_person.usda')
            if os.path.exists(default_usd):
                usd_arg = default_usd
    if usd_arg and usd_arg.lower() != 'none':
        usd_file = os.path.expanduser(usd_arg)
        if os.path.exists(usd_file):
            usd_person.import_usd_to_person_target(
                usd_path=usd_file,
                model_dir=person_model_dir,
                target_height=float(g('person_height')),
            )

    gz_args = f'-r -v 2 {os.path.join(share, "sim", "worlds", g("world"))}'
    if g('gui').lower() != 'true':
        gz_args = '-s ' + gz_args
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items())

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
               parameters=[{'use_sim_time': True, 'robot_description': desc}])
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description', '-name', 'lite6', '-x', '0', '-y', '0', '-z', '0'],
                 parameters=[{'use_sim_time': True}])
    person_joints = usd_person.JOINTS
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                  parameters=[{'use_sim_time': True}],
                  arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                             '/sim_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                             '/sim_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                             '/sim_depth/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                             *[f'/person_target/{j}@std_msgs/msg/Float64]gz.msgs.Double'
                               for j in person_joints]])

    spawners = [Node(package='controller_manager', executable='spawner', output='screen',
                     arguments=[c, '--controller-manager', '/controller_manager'])
                for c in ('joint_state_broadcaster', 'lite6_traj_controller')]
    goal = ('{trajectory: {joint_names: [joint1, joint2, joint3, joint4, joint5, joint6], '
            f'points: [{{positions: {READY}, time_from_start: {{sec: 4}}}}]}}}}')
    go_ready = ExecuteProcess(
        cmd=['ros2', 'action', 'send_goal', '/lite6_traj_controller/follow_joint_trajectory',
             'control_msgs/action/FollowJointTrajectory', goal],
        output='screen')

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('xarm_moveit_config'), 'launch', '_robot_moveit_common2.launch.py')),
        launch_arguments={'prefix': '', 'attach_to': 'world', 'attach_xyz': '"0 0 0"',
                          'attach_rpy': '"0 0 0"', 'no_gui_ctrl': 'false', 'show_rviz': g('rviz'),
                          'use_sim_time': 'true', 'moveit_config_dump': moveit_config_dump}.items())

    # Raíz de la cámara -> óptico (como lo haría un driver real); sin padre hasta calibrar.
    cam_tf = Node(package='tf2_ros', executable='static_transform_publisher', name='sim_camera_tf',
                  arguments=['--x', '0', '--y', '0', '--z', '0',
                             '--roll', str(sim_camera.LINK_TO_OPTICAL_RPY[0]), '--pitch', '0',
                             '--yaw', str(sim_camera.LINK_TO_OPTICAL_RPY[2]),
                             '--frame-id', cam['camera_root_frame'],
                             '--child-frame-id', cam['optical_frame']],
                  parameters=[{'use_sim_time': True}])
    detector = Node(package='lite6_handeye', executable='charuco_detector', name='charuco_detector',
                    output='screen',
                    parameters=[node_params(params_file, 'charuco_detector', {
                        'image_topic': cam['image_topic'], 'camera_info_topic': cam['camera_info_topic'],
                        'use_sim_time': True})])

    return [
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.pathsep.join(filter(None, [
            os.path.join(share, 'sim', 'models'), os.environ.get('GZ_SIM_RESOURCE_PATH', '')]))),
        gazebo, rsp, spawn, bridge, cam_tf, detector, moveit,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[spawners[0]])),
        RegisterEventHandler(OnProcessExit(target_action=spawners[0], on_exit=[spawners[1]])),
        RegisterEventHandler(OnProcessExit(target_action=spawners[1], on_exit=[go_ready])),
    ]


def generate_launch_description():
    share = get_package_share_directory('lite6_handeye')
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='GUI de Gazebo'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('world', default_value='handeye_table.sdf'),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'lite6_handeye.yaml')),
        # Error de montaje respecto del CAD (x y z [m] roll pitch yaw [rad]) -> X verdadero
        DeclareLaunchArgument('cam_error', default_value='0.008 -0.005 0.006 0.035 -0.052 0.070'),
        DeclareLaunchArgument('ground_truth_file',
                              default_value=os.path.expanduser('~/.ros/lite6_handeye/sim_ground_truth.yaml')),
        # astra_pro_plus: FOV y rango de profundidad del datasheet; generic: cámara anterior (H 86°)
        DeclareLaunchArgument('camera_model', default_value='astra_pro_plus'),
        DeclareLaunchArgument(
            'person_usd', default_value='',
            description='Ruta a archivo .usd/.usda/.usdc/.usdz de persona (vacío: auto en ~/.ros/lite6_handeye/person_usd/)'),
        DeclareLaunchArgument('person_height', default_value='1.58',
                              description='Estatura objetivo de la persona USD en metros'),
        OpaqueFunction(function=_setup),
    ])
