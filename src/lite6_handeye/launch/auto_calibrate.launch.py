"""Calibración automática (requiere bringup.launch.py corriendo).

  Simulación (valida poses, sin cámara):
    ros2 launch lite6_handeye auto_calibrate.launch.py sim:=true
  Real, generando poses alrededor del tablero detectado:
    ros2 launch lite6_handeye auto_calibrate.launch.py camera:=zed
  Real, reproduciendo las poses validadas en simulación:
    ros2 launch lite6_handeye auto_calibrate.launch.py camera:=zed \
        poses_file:=~/.ros/lite6_handeye/lite6_zed_calib_poses.yaml

  Gazebo (sim_gazebo.launch.py corriendo):
    ros2 launch lite6_handeye auto_calibrate.launch.py camera:=sim

Iniciar: ros2 service call /handeye_auto_calibrator/start std_srvs/srv/Trigger
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from lite6_handeye.launch_utils import node_params
from lite6_handeye.presets import CAMERAS


def _setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    sim = g('sim').lower() == 'true'
    cam = CAMERAS[g('camera')]
    overrides = {
        'camera_root_frame': cam['camera_root_frame'],
        'capture': not sim,
        'use_board_pose_base': sim,
        'poses_file': os.path.expanduser(g('poses_file')),
        'n_poses': int(g('n_poses')),
        'use_sim_time': g('camera') == 'sim',
    }
    return [Node(package='lite6_handeye', executable='auto_calibrator',
                 name='handeye_auto_calibrator', output='screen',
                 parameters=[node_params(g('params_file'), 'handeye_auto_calibrator', overrides)])]


def generate_launch_description():
    share = FindPackageShare('lite6_handeye').find('lite6_handeye')
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='false'),
        DeclareLaunchArgument('camera', default_value='zed', description='zed | orbbec | sim'),
        DeclareLaunchArgument('poses_file', default_value=''),
        DeclareLaunchArgument('n_poses', default_value='20'),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'lite6_handeye.yaml')),
        OpaqueFunction(function=_setup),
    ])
