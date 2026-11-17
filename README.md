# lite6_zed — Calibración Hand-Eye (Eye-in-Hand) en ROS 2 Jazzy

UFactory Lite 6 + cámara ZED u Orbbec montada en la brida (`link_eef`).
Se calibra con un tablero ChArUco grabado en aluminio.

Paquete: `src/lite6_handeye` (ament_python).

| Ejecutable / launch | Función |
| --- | --- |
| `generate_board` | Genera el ChArUco en SVG vectorial 1:1 (mm) para el láser, más un PNG |
| `charuco_detector` | Imagen + `camera_info` → TF `óptico → charuco_board` y `~/board_pose`, con `~/debug_image` |
| `auto_calibrator` | Recorre 15–25 poses con MoveIt 2, captura muestras, resuelve AX = XB y guarda el YAML |
| `solve_offline` | Vuelve a resolver la calibración a partir de las muestras guardadas |
| `person_tracker` | Detecta personas en RGB y ejecuta seguimiento Visual Servoing 6-DOF usando el TF calibrado |
| `bringup.launch.py` | Lanza el robot (real o simulado) con MoveIt, la cámara, el detector y, si se indica, el TF calibrado |
| `auto_calibrate.launch.py` | Lanza la calibración automática (en simulación o en el robot real) |
| `easy_handeye2.launch.py` | Alternativa manual con GUI, con los frames corregidos |
| `publish_calibration.launch.py` | Publica el resultado como TF estático |
| `person_tracker.launch.py` | Lanza la detección y seguimiento 6-DOF de personas |

## Diferencias respecto del plan original (correcciones)

1. **El TF se publica `link_eef → raíz de la cámara`, no hacia el frame óptico.** El driver ya publica
   `zed_camera_link → … → zed_left_camera_frame_optical`, y un frame no puede tener dos padres. El
   calibrador resuelve `X = link_eef → óptico` y publica `X · (raíz → óptico)⁻¹`.
2. **El detector es propio (`charuco_detector`).** `aruco_ros` no estima la pose de tableros ChArUco, y
   `opencv_cam` es un driver de cámara.
3. **En la ZED hay que usar `publish_tf:=false`.** Si no, la odometría publica `odom → zed_camera_link`,
   lo que choca con la calibración. `bringup.launch.py` ya lo fija.
4. **`link_eef` es la brida.** La cámara va montada directamente mediante su soporte en `link_eef` (sin herramienta adicional).
5. **El solver está en numpy/scipy.** Usa Park-Martin con refinamiento no lineal y rechazo de outliers
   (MAD). No depende de `cv2.calibrateHandEye`, que no está en OpenCV 5 para Python. El código ArUco
   funciona con OpenCV 4.6 (apt de Ubuntu 24.04) y con OpenCV ≥ 4.7.

## 1. Workspace

En hosts que no son Ubuntu (Arch/Omarchy), usa el entorno Docker: `docker/README.md`.


```bash
mkdir -p ~/lite6_ws/src && cd ~/lite6_ws
cp -r <este repo>/src/lite6_handeye src/
vcs import --recursive src < <este repo>/lite6_zed.repos   # borrar del .repos la cámara que no se use
sudo apt install python3-opencv python3-scipy python3-yaml
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

La ZED requiere ZED SDK + CUDA. La Orbbec requiere las reglas udev del SDK (ver su README).

## 2. Tablero ChArUco

```bash
ros2 run lite6_handeye generate_board --out charuco_5x7 \
    --squares-x 5 --squares-y 7 --square-length 0.030 --marker-length 0.022 --dictionary DICT_4X4_50
```

El tablero por defecto ya está generado en `boards/` (170 × 230 mm con un margen de 10 mm).

- Importa el SVG a LightBurn/LaserGRBL a **escala 1:1** (las unidades son mm). Negro = zona grabada.
- En aluminio anodizado oscuro, el láser deja **más claro** el metal grabado. En ese caso usa `--invert`:
  así se graban las zonas blancas y el anodizado queda como negro. El metal desnudo produce reflejos
  especulares, así que haz una prueba en un retazo y verifica la detección con una foto antes de grabar
  la placa final.
- `squares_y` debe ser impar. OpenCV 4.6 y ≥ 4.7 dibujan distinto los tableros con filas pares.
- **Después de grabar**, mide con el calibrador `square_length` y `marker_length` reales y escríbelos en
  `config/lite6_handeye.yaml` (`board.*`). Un error del 1 % en la escala produce un error del 1 % en la
  distancia.

## 3. Intrínsecos

El detector usa el `camera_info` del driver: imagen rectificada en la ZED y K + D de fábrica en la Orbbec.
Recalibra con `camera_calibration` solo si ves errores de reproyección > 1 px en `~/debug_image`.

## 4. Simulación previa (MoveIt fake, sin cámara)

```bash
ros2 launch lite6_handeye bringup.launch.py fake:=true camera:=none
ros2 launch lite6_handeye auto_calibrate.launch.py sim:=true
ros2 service call /handeye_auto_calibrator/start std_srvs/srv/Trigger
```

- Ajusta primero en `config/lite6_handeye.yaml`:
  - `x_prior_xyz_rpy`: estimación CAD de `link_eef → óptico`.
  - `board_pose_base`: dónde estará el tablero.
  - `table.*`: la mesa.
  - Los argumentos `tool_height/length/width` de bringup: la caja de colisión del soporte + cámara.
- La mesa se agrega como `CollisionObject` y el soporte + cámara como `add_other_geometry`.
- Las poses alcanzables se guardan en `~/.ros/lite6_handeye/lite6_zed_calib_poses.yaml` (posiciones
  articulares).
- Revisa en RViz que no haya singularidades ni tirones del cable.
- **Pose inicial:** parte de una pose libre de colisión. Con todas las articulaciones en 0, el soporte
  toca la base. En simulación:
  ```bash
  ros2 action send_goal /lite6_traj_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
    "{trajectory: {joint_names: [joint1, joint2, joint3, joint4, joint5, joint6], points: [{positions: [0.0, 0.2, 1.0, 0.0, 0.8, 0.0], time_from_start: {sec: 4}}]}}"
  ```
  > ⚠️ En el robot real no uses este comando: se salta la verificación de colisiones de MoveIt. Mueve el
  > robot con UFactory Studio o desde RViz.
- Cada objetivo se resuelve con IK sembrada en la configuración actual (mínimo giro de muñeca). El log
  muestra el rango recorrido por `joint4` y `joint6`. Si el cable de la cámara lo exige, limítalo con
  `wrist_window_deg`.

## 5. Calibración real

1. Fija el tablero con prensas o cinta doble faz. **No debe moverse** durante toda la captura.
2. Lanza el robot y la cámara:
   ```bash
   ros2 launch lite6_handeye bringup.launch.py robot_ip:=<IP> camera:=zed     # o camera:=orbbec
   ```
3. Mueve el robot (en modo manual o desde RViz) hasta que el tablero se vea centrado. Comprueba la
   detección con `rqt_image_view /charuco_detector/debug_image`.
4. Lanza el calibrador y arráncalo:
   ```bash
   ros2 launch lite6_handeye auto_calibrate.launch.py camera:=zed \
       poses_file:=~/.ros/lite6_handeye/lite6_zed_calib_poses.yaml   # omitir para generar poses nuevas
   ros2 service call /handeye_auto_calibrator/start std_srvs/srv/Trigger
   ```

   > ⚠️ El robot real se mueve solo, al 10 % de velocidad (`velocity_scaling`). Ten el paro de emergencia
   > a mano. Las poses generadas a partir de la detección real pueden diferir de las simuladas; para
   > ejecutar exactamente lo validado en simulación, usa `poses_file`.

5. En cada pose, el calibrador espera `settle_time` (1 s), promedia 10 detecciones y descarta la muestra
   si hubo vibración o movimiento.
6. Al terminar vuelve a la pose inicial, resuelve y escribe:
   - `~/.ros/lite6_handeye/lite6_zed_calib.yaml`: resultado, métricas y el comando
     `static_transform_publisher`.
   - `~/.ros/lite6_handeye/lite6_zed_calib_samples.yaml`: muestras crudas. Para volver a resolver:
     `ros2 run lite6_handeye solve_offline <archivo> --method park|refine`.

**Criterio de calidad:** `metrics.trans_rms_mm` es la dispersión de la posición del tablero en
`link_base` estimada desde cada pose. Una buena calibración da < 2 mm RMS y < 0.3° RMS. Si es mayor,
revisa las medidas del tablero, la rigidez del soporte y la variación angular de las poses.

**Alternativa manual con GUI:** `ros2 launch lite6_handeye easy_handeye2.launch.py camera:=zed`. Usa
`tracking_base_frame` = raíz de la cámara y no se puede usar a la vez que `calibration_file:=`.

## 6. Uso: publicar el TF calibrado + seguimiento 6-DOF de personas

> Robot real con Orbbec: arquitectura completa, parámetros afinados y cómo lanzar la app web en
> [`docs/ARQUITECTURA_SEGUIMIENTO.md`](docs/ARQUITECTURA_SEGUIMIENTO.md).

```bash
ros2 launch lite6_handeye bringup.launch.py robot_ip:=<IP> camera:=orbbec start_detector:=false \
    calibration_file:=~/.ros/lite6_handeye/lite6_zed_calib.yaml
ros2 launch lite6_handeye person_tracker.launch.py camera:=orbbec
```

Pasos de `person_tracker`:

1. Lleva el brazo a la postura inicial de vigilancia horizontal (`WATCH_JOINTS`).
2. Detecta a la persona en la imagen RGB (`PersonDetector`: torso/cabeza humanoide, HOG o modelo ONNX).
3. Reconstruye la posición 3D en `link_base` usando los intrínsecos `K` y la transformación Hand-Eye calibrada (`link_eef → óptico`) y la suaviza con un filtro exponencial (`smoothing`).
4. Calcula la pose 6-DOF del efector (`compute_tracking_pose_6dof`), que **traslada** la cámara en 3D (acimut, radio y altura) y **reorienta** su eje óptico (`+z`) directamente hacia la persona manteniendo el horizonte nivelado.
5. Resuelve la cinemática inversa 6-DOF (`lite6_ik_6dof`) sembrada en la configuración actual y envía trayectorias suaves a `/lite6_traj_controller/follow_joint_trajectory`.
6. Publica `/person_tracker/debug_image`, `/person_tracker/target_point` y `/person_tracker/target_eef_pose`.

## Simulación en Gazebo (ground truth + seguimiento 6-DOF)

```bash
ros2 launch lite6_handeye sim_gazebo.launch.py            # Lite 6 + cámara con X verdadero + tablero + person_target
ros2 launch lite6_handeye auto_calibrate.launch.py camera:=sim
ros2 service call /handeye_auto_calibrator/start std_srvs/srv/Trigger
ros2 run lite6_handeye compare_gt                          # error vs ground truth
ros2 launch lite6_handeye publish_calibration.launch.py
ros2 launch lite6_handeye person_tracker.launch.py camera:=sim
```

Resultado de referencia:
- Calibración: 0.019 mm y 0.006°.
- Seguimiento 6-DOF de persona en movimiento: error angular medio 0.28° (máx. 0.61°), traslación 3D activa ($\Delta y = 161\text{ mm}, \Delta z = 40\text{ mm}$).

## Pruebas

```bash
cd src/lite6_handeye && python3 -m pytest test
```

Cubren:

- Solver con datos sintéticos (exacto, con ruido, con outliers y caso degenerado).
- Layout del tablero idéntico a `CharucoBoard.generateImage`.
- Recuperación de pose desde una imagen proyectada.
- Dimensiones del SVG y geometría de las poses.
- Ida y vuelta de E/S y CLI.
