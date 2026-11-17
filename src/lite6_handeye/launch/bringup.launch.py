"""Robot Lite 6 (MoveIt 2) + cámara + detector ChArUco [+ calibración publicada].

  Simulación:  ros2 launch lite6_handeye bringup.launch.py fake:=true camera:=none
  Real (ZED):  ros2 launch lite6_handeye bringup.launch.py robot_ip:=192.168.1.xxx camera:=zed
  Real + TF calibrado:
               ros2 launch lite6_handeye bringup.launch.py robot_ip:=... camera:=zed \
                   calibration_file:=~/.ros/lite6_handeye/lite6_zed_calib.yaml
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from lite6_handeye.launch_utils import node_params
from lite6_handeye.presets import CAMERAS


def _include(pkg, rel, args):
    path = os.path.join(FindPackageShare(pkg).find(pkg), rel)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=list(args.items()))


def _setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    fake = g('fake').lower() == 'true'
    camera = g('camera')
    params = g('params_file')
    actions = []

    robot_args = {
        'dof': '6', 'robot_type': 'lite', 'hw_ns': 'ufactory', 'no_gui_ctrl': 'false',
        # Volumen del soporte + cámara como colisión en link_eef
        'add_other_geometry': 'true', 'geometry_type': 'box',
        'geometry_height': g('tool_height'), 'geometry_length': g('tool_length'),
        'geometry_width': g('tool_width'),
    }
    if fake:
        actions.append(_include('xarm_moveit_config', 'launch/_robot_moveit_fake.launch.py', robot_args))
    else:
        robot_args['robot_ip'] = g('robot_ip')
        actions.append(_include('xarm_moveit_config', 'launch/_robot_moveit_realmove.launch.py',
                                robot_args))

    if camera == 'zed':
        actions.append(_include('zed_wrapper', 'launch/zed_camera.launch.py', {
            'camera_model': g('zed_camera_model'), 'camera_name': 'zed',
            'publish_tf': 'false', 'publish_map_tf': 'false', 'publish_urdf': 'true'}))
    elif camera == 'orbbec':
        try:
            FindPackageShare('orbbec_camera').find('orbbec_camera')
            actions.append(_include('orbbec_camera', 'launch/' + g('orbbec_launch'), {
                'camera_name': 'camera', 'enable_point_cloud': 'true', 'time_domain': 'system'}))
        except Exception:
            c = CAMERAS['orbbec']
            actions.append(Node(
                package='lite6_handeye', executable='orbbec_uvc', name='orbbec_uvc',
                output='screen',
                parameters=[{'optical_frame': c['optical_frame'],
                             'image_topic': c['image_topic'],
                             'camera_info_topic': c['camera_info_topic']}]))
            actions.append(Node(
                package='tf2_ros', executable='static_transform_publisher', name='orbbec_optical_tf',
                arguments=['--x', '0', '--y', '0', '--z', '0',
                           '--roll', '-1.5707963267948966', '--pitch', '0',
                           '--yaw', '-1.5707963267948966',
                           '--frame-id', c['camera_root_frame'],
                           '--child-frame-id', c['optical_frame']]))
    elif camera != 'none':
        raise RuntimeError(f'camera desconocida: {camera} (zed | orbbec | none)')

    if camera in CAMERAS and g('start_detector').lower() == 'true':
        c = CAMERAS[camera]
        actions.append(Node(package='lite6_handeye', executable='charuco_detector',
                            name='charuco_detector', output='screen',
                            parameters=[node_params(params, 'charuco_detector', {
                                'image_topic': c['image_topic'],
                                'camera_info_topic': c['camera_info_topic']})]))

    calib = g('calibration_file')
    if calib:
        pub_args = {'calibration_file': calib}
        if camera in CAMERAS:
            pub_args['child_frame'] = CAMERAS[camera]['camera_root_frame']
        actions.append(_include('lite6_handeye', 'launch/publish_calibration.launch.py',
                                pub_args))
    return actions


def generate_launch_description():
    share = FindPackageShare('lite6_handeye').find('lite6_handeye')
    return LaunchDescription([
        DeclareLaunchArgument('fake', default_value='false', description='MoveIt fake (simulación)'),
        DeclareLaunchArgument('robot_ip', default_value='192.168.1.153'),
        DeclareLaunchArgument('camera', default_value='orbbec', description='zed | orbbec | none'),
        DeclareLaunchArgument('zed_camera_model', default_value='zed2i'),
        DeclareLaunchArgument('orbbec_launch', default_value='gemini_330_series.launch.py'),
        DeclareLaunchArgument('start_detector', default_value='true'),
        DeclareLaunchArgument('calibration_file', default_value=''),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'lite6_handeye.yaml')),
        # Caja de colisión del soporte + cámara (m): height = a lo largo de z de link_eef
        DeclareLaunchArgument('tool_height', default_value='0.05'),
        DeclareLaunchArgument('tool_length', default_value='0.18'),
        DeclareLaunchArgument('tool_width', default_value='0.06'),
        OpaqueFunction(function=_setup),
    ])
