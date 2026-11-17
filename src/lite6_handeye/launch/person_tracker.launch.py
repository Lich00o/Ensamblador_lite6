"""Detección de personas y seguimiento 6-DOF con cámara Eye-in-Hand.

  Gazebo (sim_gazebo.launch.py corriendo):
    ros2 launch lite6_handeye person_tracker.launch.py camera:=sim

  Robot real (bringup.launch.py corriendo):
    ros2 launch lite6_handeye person_tracker.launch.py camera:=orbbec
    ros2 launch lite6_handeye person_tracker.launch.py camera:=zed

Salidas:
  /person_tracker/debug_image       (sensor_msgs/Image)
  /person_tracker/target_point      (geometry_msgs/PointStamped en link_base)
  /person_tracker/target_eef_pose   (geometry_msgs/PoseStamped en link_base)
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
    cam = CAMERAS[g('camera')]
    overrides = {
        'image_topic': cam['image_topic'],
        'camera_info_topic': cam['camera_info_topic'],
        'optical_frame': cam['optical_frame'],
        'calibration_file': os.path.expanduser(g('calibration_file')),
        'enable_motion': g('enable_motion').lower() == 'true',
        'use_sim_time': g('camera') == 'sim',
        'robot_states_topic': '' if g('camera') == 'sim' else '/ufactory/robot_states',
    }
    return [Node(
        package='lite6_handeye',
        executable='person_tracker',
        name='person_tracker',
        output='screen',
        parameters=[node_params(g('params_file'), 'person_tracker', overrides)],
    )]


def generate_launch_description():
    share = FindPackageShare('lite6_handeye').find('lite6_handeye')
    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='sim', description='zed | orbbec | sim'),
        DeclareLaunchArgument(
            'calibration_file',
            default_value='~/.ros/lite6_handeye/lite6_zed_calib.yaml',
            description='Archivo YAML con la calibración Hand-Eye',
        ),
        DeclareLaunchArgument('enable_motion', default_value='true'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(share, 'config', 'lite6_handeye.yaml'),
        ),
        OpaqueFunction(function=_setup),
    ])
