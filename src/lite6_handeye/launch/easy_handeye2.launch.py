"""Alternativa manual (GUI rqt) con easy_handeye2, frames corregidos.

  ros2 launch lite6_handeye easy_handeye2.launch.py camera:=zed

tracking_base_frame = raíz del árbol TF de la cámara (no el frame óptico):
easy_handeye2 publica un TF provisional link_eef -> tracking_base_frame y el
resultado se publica sobre ese mismo par. NO lanzar a la vez con
calibration_file:= en bringup (dos padres para el mismo frame).

Resultado: ~/.ros2/easy_handeye2/calibrations/<name>.calib
Publicar:  ros2 launch easy_handeye2 publish.launch.py name:=<name>
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

from lite6_handeye.presets import CAMERAS


def _setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    cam = CAMERAS[g('camera')]
    path = os.path.join(FindPackageShare('easy_handeye2').find('easy_handeye2'),
                        'launch', 'calibrate.launch.py')
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=[
        ('name', g('name')),
        ('calibration_type', 'eye_in_hand'),
        ('robot_base_frame', 'link_base'),
        ('robot_effector_frame', 'link_eef'),
        ('tracking_base_frame', cam['camera_root_frame']),
        ('tracking_marker_frame', 'charuco_board'),
    ])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='zed', description='zed | orbbec | sim'),
        DeclareLaunchArgument('name', default_value='lite6_zed_calib'),
        OpaqueFunction(function=_setup),
    ])
