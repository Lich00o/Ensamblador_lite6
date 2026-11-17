"""Cámara RGB-D simulada (Gazebo Harmonic) montada en link_eef con X verdadero conocido.

Frames:
  link_eef -> sim_camera_gt            (URDF, ground truth; NO es el frame que publica la calibración)
  sim_camera_link -> sim_camera_optical_frame   (TF estático aparte, sin padre, como un driver real)
La calibración publica link_eef -> sim_camera_link y se compara contra link_eef -> sim_camera_gt.

Convención gz: la cámara mira a lo largo de +x del link (igual que sim_camera_link);
el frame óptico (z adelante, y abajo) es link * rpy(-pi/2, 0, -pi/2).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import T_from_xyz_rpy, inv_T

LINK_TO_OPTICAL_RPY = (-np.pi / 2, 0.0, -np.pi / 2)
T_LINK_OPTICAL = T_from_xyz_rpy([0.0, 0.0, 0.0], LINK_TO_OPTICAL_RPY)


def true_transforms(prior_xyz_rpy, error_xyz_rpy):
    """X verdadero = prior (CAD) perturbado. Devuelve (T_eef_optical, T_eef_link)."""
    p = np.asarray(prior_xyz_rpy, float)
    e = np.asarray(error_xyz_rpy, float)
    X = T_from_xyz_rpy(p[:3], p[3:]) @ T_from_xyz_rpy(e[:3], e[3:])
    return X, X @ inv_T(T_LINK_OPTICAL)


# Orbbec Astra Pro Plus (datasheet): RGB 1920x1080 H66.1°; profundidad 640x480 H58.4°, 0.6-8 m.
# El RGB se simula a 1280x720 con el mismo FOV (menos carga; los intrínsecos escalan).
ASTRA_PRO_PLUS = dict(width=1280, height=720, hfov=np.deg2rad(66.1),
                      depth_width=640, depth_height=480, depth_hfov=np.deg2rad(58.4))

# Caja de colisión de la cámara en link_eef (m): envolvente del STEP posicionado (sim/meshes).
CAMERA_BOX_CENTER = (0.0, -0.0725, 0.016)
CAMERA_BOX_SIZE = (0.166, 0.036, 0.059)


def body_snippet(mesh_uri, parent='link_eef', link='camera_body'):
    """Cuerpo de la cámara (malla visual en el marco del flange + caja de colisión)."""
    c = ' '.join(f'{v:.4f}' for v in CAMERA_BOX_CENTER)
    b = ' '.join(f'{v:.4f}' for v in CAMERA_BOX_SIZE)
    return f"""
  <!-- lite6_handeye: cuerpo de la cámara (CAD del soporte, marco = link_eef) -->
  <link name="{link}">
    <!-- 310 g (datasheet); CoG del STEP; inercia de caja 165x30x40 mm -->
    <inertial>
      <origin xyz="0 -0.0701 0.025" rpy="0 0 0"/>
      <mass value="0.31"/><inertia ixx="6.5e-5" ixy="0" ixz="0" iyy="7.4e-4" iyz="0" izz="7.3e-4"/>
    </inertial>
    <visual>
      <geometry><mesh filename="{mesh_uri}"/></geometry>
      <material name="camera_black"><color rgba="0.12 0.12 0.12 1"/></material>
    </visual>
    <collision>
      <origin xyz="{c}" rpy="0 0 0"/>
      <geometry><box size="{b}"/></geometry>
    </collision>
  </link>
  <joint name="{link}_joint" type="fixed">
    <parent link="{parent}"/>
    <child link="{link}"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
"""


def urdf_snippet(T_eef_link, parent='link_eef', width=1280, height=720, hfov=1.5, rate=15.0,
                 image_noise=0.005, topic='sim_camera', optical_frame='sim_camera_optical_frame',
                 depth_topic='sim_depth', depth_width=640, depth_height=360, depth_rate=5.0,
                 depth_hfov=None):
    xyz = ' '.join(f'{v:.6f}' for v in T_eef_link[:3, 3])
    rpy = ' '.join(f'{v:.6f}' for v in Rotation.from_matrix(T_eef_link[:3, :3]).as_euler('xyz'))
    depth_hfov = hfov if depth_hfov is None else depth_hfov
    return f"""
  <!-- lite6_handeye: cámara simulada (ground truth) -->
  <link name="sim_camera_gt">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <joint name="sim_camera_gt_joint" type="fixed">
    <parent link="{parent}"/>
    <child link="sim_camera_gt"/>
    <origin xyz="{xyz}" rpy="{rpy}"/>
  </joint>
  <gazebo reference="sim_camera_gt">
    <sensor name="sim_camera" type="rgbd_camera">
      <always_on>true</always_on>
      <update_rate>{rate}</update_rate>
      <topic>{topic}</topic>
      <gz_frame_id>{optical_frame}</gz_frame_id>
      <camera>
        <horizontal_fov>{hfov}</horizontal_fov>
        <image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>
        <clip><near>0.05</near><far>3.0</far></clip>
        <noise><type>gaussian</type><mean>0.0</mean><stddev>{image_noise}</stddev></noise>
        <depth_camera><clip><near>0.1</near><far>3.0</far></clip></depth_camera>
      </camera>
    </sensor>
    <!-- Nube de puntos en resolución reducida: 1280x720 a 15 Hz son ~30 MB por nube,
         demasiado para deserializar en Python (atrasa TF). -->
    <sensor name="sim_depth" type="rgbd_camera">
      <always_on>true</always_on>
      <update_rate>{depth_rate}</update_rate>
      <topic>{depth_topic}</topic>
      <gz_frame_id>{optical_frame}</gz_frame_id>
      <camera>
        <horizontal_fov>{depth_hfov}</horizontal_fov>
        <image><width>{depth_width}</width><height>{depth_height}</height><format>R8G8B8</format></image>
        <!-- Sin rango mínimo del sensor aquí: gz recorta y "ve" lo que hay detrás. -->
        <clip><near>0.05</near><far>3.0</far></clip>
        <depth_camera><clip><near>0.05</near><far>3.0</far></clip></depth_camera>
      </camera>
    </sensor>
  </gazebo>
"""


def srdf_disable(other_links, link='camera_body'):
    return ''.join(f'  <disable_collisions link1="{link}" link2="{o}" reason="Adjacent"/>\n'
                   for o in other_links)


def inject(robot_description, snippet):
    i = robot_description.rfind('</robot>')
    if i < 0:
        raise ValueError('robot_description sin </robot>')
    return robot_description[:i] + snippet + robot_description[i:]
