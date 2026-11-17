"""Publica el resultado (YAML de auto_calibrator) como TF estático.

  ros2 launch lite6_handeye publish_calibration.launch.py \
      calibration_file:=~/.ros/lite6_handeye/lite6_zed_calib.yaml
"""

import os

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    path = os.path.expanduser(LaunchConfiguration('calibration_file').perform(context))
    child_override = LaunchConfiguration('child_frame').perform(context).strip()
    with open(path) as f:
        p = yaml.safe_load(f)['publish']
    t, r = p['translation'], p['rotation']
    child = child_override if child_override else p['child_frame']
    args = ['--x', str(t['x']), '--y', str(t['y']), '--z', str(t['z']),
            '--qx', str(r['x']), '--qy', str(r['y']), '--qz', str(r['z']), '--qw', str(r['w']),
            '--frame-id', p['parent_frame'], '--child-frame-id', child]
    return [Node(package='tf2_ros', executable='static_transform_publisher',
                 name='handeye_calibration_publisher', arguments=args, output='screen')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('calibration_file',
                              default_value='~/.ros/lite6_handeye/lite6_zed_calib.yaml'),
        DeclareLaunchArgument('child_frame', default_value=''),
        OpaqueFunction(function=_setup),
    ])
