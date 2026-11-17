"""Tópicos y frames por cámara.

camera_root_frame: raíz del árbol TF del driver. La calibración se publica
link_eef -> camera_root_frame (nunca al frame óptico: ya tiene padre).
"""

CAMERAS = {
    # zed-ros2-wrapper (master, camera_name:=zed). publish_tf:=false obligatorio:
    # si no, el driver publica odom -> zed_camera_link y choca con link_eef -> zed_camera_link.
    'zed': {
        'image_topic': '/zed/zed_node/rgb/color/rect/image',
        'camera_info_topic': '/zed/zed_node/rgb/color/rect/camera_info',
        'cloud_topic': '/zed/zed_node/point_cloud/cloud_registered',
        'camera_root_frame': 'zed_camera_link',
        'optical_frame': 'zed_left_camera_frame_optical',
    },
    # Gazebo (sim_gazebo.launch.py). cloud_frame: gz publica los puntos en el frame del
    # link (x adelante) aunque frame_id diga óptico; sim_camera_link tiene esa convención.
    'sim': {
        'image_topic': '/sim_camera/image',
        'camera_info_topic': '/sim_camera/camera_info',
        'cloud_topic': '/sim_depth/points',          # sensor de 640x360 (ver sim_camera.py)
        'cloud_frame_override': 'sim_camera_link',
        # gz no invalida lo que está bajo el mínimo del sensor (lo recorta y ve detrás):
        # se filtra en el consumidor, como lo haría el sensor real (Astra Pro Plus, 0.6 m)
        'cloud_min_range': 0.6,
        'camera_root_frame': 'sim_camera_link',
        'optical_frame': 'sim_camera_optical_frame',
    },
    # OrbbecSDK_ROS2 v2-main (camera_name:=camera)
    'orbbec': {
        'image_topic': '/camera/color/image_raw',
        'camera_info_topic': '/camera/color/camera_info',
        'cloud_topic': '/camera/depth/points',
        'cloud_min_range': 0.6,                      # Astra Pro Plus: 0.6-8 m
        'camera_root_frame': 'camera_link',
        'optical_frame': 'camera_color_optical_frame',
    },
}
