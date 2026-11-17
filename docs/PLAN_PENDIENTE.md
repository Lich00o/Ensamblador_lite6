# Plan pendiente — validar sin robot y preparar el día del robot real

Estado al 2026-10-22: el paquete `src/lite6_handeye` está implementado y probado con datos sintéticos
(15 pruebas unitarias, nodos con rclpy, calibrador de punta a punta con MoveIt simulado por stub).
Este documento lista lo que falta, en orden. Cada fase tiene un criterio de aceptación.

---

## Fase 0 — Entorno Docker (Ubuntu 24.04 + Jazzy)

> **✅ COMPLETADA 2026-11-09.** Ver `docker/README.md`. Resultado:
> - Imagen `lite6_zed:jazzy` (7.4 GB). `xarm_ros2` + `easy_handeye2` + `lite6_handeye` compilan.
> - `colcon test`: 15/15 con OpenCV 4.6 (la ruta 4.6 quedó validada).
> - Los 5 launch cargan (`--show-args`).
> - OpenGL 4.6 y RViz funcionan con AMD y con NVIDIA.
>
> Cambios respecto de lo planeado:
> - `rosdep` reemplazado por `docker/install_deps.py`: la red bloquea `raw.githubusercontent.com`.
> - Se eliminó un helper de conversión de frames para OpenCV 4.6, que resultó innecesario: el frame del
>   tablero es el mismo en 4.6 y en ≥ 4.7.
> - `setup.py` usa `extras_require={'test': ['pytest']}`, para que colcon corra pytest.

Motivo: la máquina de desarrollo es Arch (Omarchy). No hay binarios de MoveIt/xarm/gz para Arch, y
`colcon build` falla por el plugin `ptr` (requiere `pkg_resources`).

Archivos a crear:
- `docker/Dockerfile`
  - Base: `osrf/ros:jazzy-desktop-full` (incluye Gazebo Harmonic + `ros_gz`).
  - `apt`: `ros-jazzy-moveit`, `ros-jazzy-gz-ros2-control`, `ros-jazzy-ros-gz`,
    `python3-opencv`, `python3-scipy`, `python3-yaml`, `python3-vcstool`, `ros-jazzy-rqt-image-view`.
  - `vcs import --recursive` de `lite6_zed.repos`, **sin** zed-ros2-wrapper ni OrbbecSDK_ROS2 (no se
    usan en simulación; la ZED requiere CUDA).
  - `rosdep install` + `colcon build --symlink-install`.
- `docker/compose.yaml`
  - Monta `src/lite6_handeye` como volumen, para editar desde el host sin reconstruir la imagen.
  - GUI en Hyprland/Wayland: `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`, `/tmp/.X11-unix` (XWayland) y
    `QT_QPA_PLATFORM=xcb` para RViz.
  - GPU: `/dev/dri` (Intel/AMD). Con NVIDIA: `nvidia-container-toolkit` + `gpus: all`.
  - `network_mode: host`, `ipc: host` (DDS).
- `docker/README.md` con build, run y `exec`.

Aceptación:
- `colcon build` sin errores.
- `colcon test --packages-select lite6_handeye` pasa (valida también la ruta OpenCV 4.6 del apt de
  Ubuntu, nunca probada).
- RViz abre desde el contenedor.

---

## Fase 1 — MoveIt fake: alcanzabilidad de las poses

> **✅ COMPLETADA 2026-11-09** (en Docker, `fake:=true`). Resultado:
> - 20/20 poses alcanzables con los nuevos valores por defecto; la reproducción con `poses_file` da
>   20/20 sin errores.
> - Poses guardadas en `docker/data/lite6_zed_calib_poses.yaml`.
> - Rango de muñeca recorrido: `joint4` 148°, `joint6` 151°.
> - `min |joint5|` = 0.29 rad: sin singularidad de muñeca.
>
> Hallazgos y correcciones:
> - **Bug en los launch:** los overrides (`use_board_pose_base`, `n_poses`) no se aplicaban si la
>   clave existía en el YAML bajo el nombre del nodo. Se corrigió con
>   `launch_utils.node_params`, que combina el YAML y los overrides en un solo dict.
> - **Inicio en colisión:** con todas las articulaciones en 0, la caja del sándwich toca `link_base`
>   (`START_STATE_IN_COLLISION`). Hay que partir de una pose libre de colisión, por ejemplo la pose
>   "ready" `[0, 0.2, 1.0, 0, 0.8, 0]` (`link_eef` en (0.264, 0, 0.281) mirando hacia abajo).
> - **Alcance:** con acimut de 360° fallaban las poses del lado del tablero opuesto a la base (más de
>   0.5 m de la base). Nuevos parámetros:
>   - `azimuth_range_deg: 100` (acimut centrado hacia la base).
>   - `distance_*`: 0.22–0.32 m.
>   - `board_pose_base` con el centro del tablero en (0.30, 0).
> - **Giros de muñeca:** con objetivos cartesianos, MoveIt elige la IK al azar y `joint6` saltaba hasta
>   304° entre poses. Ahora:
>   - IK sembrada con la configuración actual (`/compute_ik`) y luego objetivo articular.
>   - Poses ordenadas por acimut.
>   - Parámetro opcional `wrist_window_deg`.
> - **Simulador de controlador UFactory** (`uf-sim`, firmware v2.4.0):
>   - Funciona: conexión del driver, servicios `xarm_api` en modo 0 y UFactory Studio.
>   - **No ejecuta el modo servo (modo 1)**, que es el que usa ros2_control, así que no sirve para
>     trayectorias de MoveIt.
>
> Pendiente para el usuario: revisión visual en RViz.

Comandos (ya implementados):
```bash
ros2 launch lite6_handeye bringup.launch.py fake:=true camera:=none
ros2 launch lite6_handeye auto_calibrate.launch.py sim:=true
ros2 service call /handeye_auto_calibrator/start std_srvs/srv/Trigger
```

Tareas:
1. Verificar que `bringup.launch.py` incluye bien `_robot_moveit_fake.launch.py` con
   `add_other_geometry` (la caja del sándwich se ve en RViz, en `link_eef`).
2. Primera prueba real de `/move_action` y `/apply_planning_scene`. Revisar los `error_code` y
   ajustar las tolerancias de `_pose_constraints` si OMPL falla mucho.
3. Ajustar en `config/lite6_handeye.yaml`: `board_pose_base` (posición real prevista del tablero),
   `distance_*`, `tilt_*` y `min_eef_height`, hasta lograr ≥ 18/20 poses alcanzables.
4. Revisar en RViz que no haya singularidades (muñeca alineada), giros de `joint6` > 300° ni
   colisiones.

Aceptación: `~/.ros/lite6_handeye/lite6_zed_calib_poses.yaml` con ≥ 15 poses.

---

## Fase 2 — Gazebo: calibración de punta a punta con ground truth

> **✅ COMPLETADA 2026-11-10** (salvo los opcionales 2.4-c y 2.5).
>
> - **Calibración** (corrida 4): error vs ground truth **0.049 mm / 0.017°** (aceptación < 1 mm / 0.2°).
>   Consistencia 0.03 mm RMS. 16/20 poses alcanzadas. Muñeca: `joint4` 146°, `joint6` 175°.
> - **Ventosa** (`docker/data/sim_scripts/grasp_test.py`): `box_a` 0.21 mm / 0.01°, `cyl_b`
>   0.09 mm / 0.01° (aceptación < 3 mm / 2°).
> - Resultados en `docker/data/`.
>
> Cambios y hallazgos:
> - `launch/sim_gazebo.launch.py`, `sim/worlds/handeye_table.sdf`, `sim/models/charuco_board`,
>   `lite6_handeye/sim_camera.py`, CLI `compare_gt` y preset `camera:=sim`.
> - Puntos de gz en el frame del link aunque `frame_id` diga óptico: se usa `cloud_frame_override`.
> - Nube de 1280×720 (~30 MB a 9 Hz): Python no alcanza a deserializarla y atrasa la TF. Se agregó un
>   segundo sensor `sim_depth` de 640×360 a 5 Hz solo para la nube. **En las cámaras reales también hay
>   que bajar la nube:**
>   - ZED: `depth.point_cloud_res: 'REDUCED'` y `depth.point_cloud_freq` ≈ 5.
>   - Orbbec: perfil de profundidad ≤ 640×480.
> - `suction_grasp`:
>   - `MultiThreadedExecutor` (antes, con `spin_thread=True`, el `TransformListener` competía por el
>     mismo nodo).
>   - Cola de nubes de profundidad 1 y `tf_timeout` de 1 s.
>   - Cara buscada con la normal restringida a `face_max_tilt_deg`; el valor por defecto pasó de 60° a
>     30°, porque antes ganaban paredes laterales o esquinas.
> - Calibrador:
>   - `CONTROL_FAILED` cerca del objetivo se acepta (`accept_joint_error_rad`).
>   - Valores por defecto `follow_azimuth: true` y `wrist_window_deg: 120`.
> - `update_rate` de ros2_control en simulación: 200 Hz (antes 1000).
>
> Opcional pendiente:
> - Comparación con `easy_handeye2.launch.py camera:=sim`.
> - Pruebas de robustez (2.5).
>
> Para retomar:
> ```bash
> docker start l6      # o: docker compose -f docker/compose.yaml run --rm sim
> ros2 launch lite6_handeye sim_gazebo.launch.py          # gui:=false rviz:=false para headless
> ```
> Notas:
> - Los scripts de espera deben buscar `calibración abortada`, no solo `abortada`: el aviso de
>   tolerancia también contiene esa palabra.
> - Para detener el calibrador, mata también `lib/lite6_handeye/auto_calibrator`. Matar solo el launch
>   deja el nodo huérfano.

### 2.1 Cámara simulada con X verdadero conocido

Archivos a crear:
- `src/lite6_handeye/urdf/sim_camera.xacro`
  - Joint fijo `link_eef → sim_camera_gt` con `xyz/rpy` = **X verdadero**. Tomar el CAD del sándwich
    y sumarle un error intencional de unos mm y grados respecto de `x_prior_xyz_rpy`.
  - Sensor gz `rgbd_camera` en `sim_camera_gt`: 1280×720, ~90° HFOV, 30 Hz, rango 0.1–3 m, ruido
    gaussiano en imagen y profundidad.
  - `<gz_frame_id>sim_camera_optical_frame</gz_frame_id>` (`<frame_id>` en versiones viejas de gz).
- **Clave:** `sim_camera_gt` NO debe ser la raíz que publica la calibración, porque si no, el TF ya
  tendría el resultado y habría dos padres. Publicar aparte un TF estático
  `sim_camera_link → sim_camera_optical_frame` (rpy `-π/2 0 -π/2`), sin padre, igual que un driver
  real. Así el calibrador publica `link_eef → sim_camera_link`, que se compara contra
  `link_eef → sim_camera_gt`.
- Para inyectar el xacro en el URDF del Lite 6 hay dos opciones; evaluarlas y elegir una:
  - (a) Usar `geometry_mesh_*` de `add_other_geometry`. No sirve para sensores.
  - (b) Un URDF propio: wrapper xacro que incluye `xarm_device_macro` + `sim_camera.xacro`, y un
    launch que reemplaza el `robot_description` del launch de xarm.

### 2.2 Mundo con tablero y objetos

Archivos a crear:
- `src/lite6_handeye/worlds/charuco_board/model.sdf`
  - Caja de 170 × 230 × 3 mm.
  - Material PBR con `albedo_map` = `boards/charuco_5x7_30mm_4x4_50.png`.
  - Estática, sobre la mesa al alcance (≈ x 0.30, y 0.0 en `link_base`).
- `src/lite6_handeye/worlds/objects/`: 2–3 cajas (tapa plana ≥ 4 × 4 cm) y 1 cilindro, para la
  ventosa.
- Spawn con `ros_gz_sim create`.

### 2.3 Bridge y preset

- `ros_gz_bridge`:
  - `/sim_camera/image`, `/sim_camera/camera_info`, `/sim_camera/points`.
  - `/clock`, con `use_sim_time:=true` en todos los nodos.
- `lite6_handeye/presets.py`: agregar `'sim'` con esos tópicos, `camera_root_frame: sim_camera_link`
  y `optical_frame: sim_camera_optical_frame`.
- `launch/sim_gazebo.launch.py`: lanza gz + robot + cámara + tablero + objetos + bridge + detector.

### 2.4 Validación

- `src/lite6_handeye/lite6_handeye/compare_gt.py` (CLI `compare_gt`): lee el YAML de resultado y el
  TF `link_eef → sim_camera_gt` y reporta el error de traslación (mm) y de rotación (°).
- Ejecutar:
  1. `auto_calibrate` con `camera:=sim`, generando poses.
  2. Repetir con `poses_file` de la Fase 1.
  3. `solve_offline --method park` vs `refine`.
- Comparar con `easy_handeye2.launch.py camera:=sim` (Tsai/Park/Horaud) como referencia
  independiente.

Aceptación:
- Calibración: error vs ground truth < 1 mm y < 0.2°; `trans_rms_mm` < 1.
- Ventosa: con el TF calibrado publicado, `suction_grasp` da el centroide de la tapa a < 3 mm del
  real y la normal a < 2°.
- Visualizar `grasp_pose`/`pregrasp_pose` en RViz y planificarlas con MoveIt (sin pick real).

### 2.5 Pruebas de robustez (opcional)

- Ruido de imagen más alto, iluminación baja y tablero parcialmente fuera de cuadro: el detector debe
  rechazar, no dar poses malas.
- Mover el tablero a mitad de la captura: `solve` debe marcar outliers o `trans_rms_mm` alto.
- Error de escala del tablero de 1 % (`square_length` mal medido): cuantificar el impacto en X. Esto
  muestra cuán crítico es medir con el calibrador vernier.

---

## Fase 2.6 — Modelo real del soporte: Astra Pro Plus + Vacuum Gripper Lite

> **✅ 2026-11-10.** CAD de `~/Work/adaptador_orbbec` (marco = flange = `link_eef`, z hacia afuera).
>
> - Mallas en `sim/meshes/`:
>   - `soporte_ventosa.stl` (cuna + tapa + ventosa) va como `other_geometry`, y `link_tcp` queda en la
>     punta de la ventosa (z = 66.1 mm).
>   - `orbbec_astra_pro_plus.stl` es el link `camera_body`: malla visual y una caja de colisión.
>     Colisiones desactivadas con `link6` y `other_geometry_link`.
> - Sensor `camera_model:=astra_pro_plus` (por defecto):
>   - RGB a 1280×720 con H 66.1°. La cámara real es 1920×1080; se usa el mismo FOV.
>   - Profundidad a 640×480 con H 58.4°, mínimo 0.6 m (datasheet).
>   - `camera_model:=generic` vuelve a la cámara anterior. `depth_near:=<m>` cambia el mínimo.
> - Lente RGB = ventana del medio (x = 12.5, y = -70 mm). Las ventanas de los extremos, separadas
>   75 mm (baseline de la Astra), son el proyector y la cámara IR.
>   - `x_prior_xyz_rpy` = `[0.0125, -0.070, 0.042, 0, 0, pi]`.
>   - `yaw = pi` supone la bisagra/USB hacia -y. **Verificar con la cámara montada.**
> - `suction_offset` = 0.0661 m (ventosa sin comprimir; medir comprimida).
> - Calibración en Gazebo: **0.016 mm / 0.006°** vs ground truth (16/20 poses, 15/16 muestras).
> - Respaldo de los resultados con la cámara genérica: `docker/data/prev_camara_generica/`.
>
> **Profundidad mínima de 0.6 m: resuelta con vistas oblicuas.** Mirando hacia abajo, el lente sube como
> máximo a 0.42–0.48 m sobre la mesa (`sim_scripts/reach_height.py`), así que la vista cenital no sirve.
> - `pose_sampler.observation_viewpoints(target, X0, dist_range=(0.62, 0.75))` genera vistas look-at
>   ordenadas de la más cenital a la más oblicua. Se elige la primera con IK válida y sin colisiones.
> - `suction_grasp` tiene los parámetros nuevos `cloud_min_range` y `cloud_max_range`. El preset
>   `orbbec`/`sim` pone 0.6. gz no invalida lo que está bajo el mínimo: lo recorta y "ve" lo que hay
>   detrás (aparecía el piso a z -0.75). Por eso el mínimo se aplica en el nodo y no en el SDF.
> - **Zona observable** (`sim_scripts/observable_map.py`, objeto a z 0.06, distancia ≥ 0.62 m): un
>   anillo con r ≳ 0.32 m desde la base. El disco interior (x < 0.30 con |y| ≤ 0.10, y x ≤ 0.20 con
>   |y| ≤ 0.20) no se puede observar: la muñeca se pliega y `link4` choca con la cámara y la ventosa.
>   Se observa con elevaciones de 48–62°.
> - `box_a` se movió a (0.36, 0.22), dentro del anillo.
> - Prueba (`sim_scripts/observe_grasp_test.py`): **ACEPTADO**. `box_a` 2.06 mm / 0.26° y `cyl_b`
>   0.13 mm / 1.07°, ambos vistos desde 0.62 m con 55–62° de elevación.
> - Para la Fase 3:
>   - Medir el mínimo real de la Astra. Si entrega datos bajo 0.6 m, bajar `cloud_min_range` agranda la
>     zona observable.
>   - Con elevaciones de 50–60°, el ruido del structured light sobre la cara superior es mayor que en
>     la sim (la sim no tiene ruido de profundidad).
> - **Área de trabajo** (`sim_scripts/radial_workspace.py`): `joint1` gira 360°, así que basta un perfil
>   radial. "Observable" = hay una vista a ≥ 0.62 m. "Agarre" = pregrasp y grasp con la ventosa
>   vertical, ambos sin colisiones.
>   - Tapa a z ≤ 0.06 m: observable y agarrable para r ≈ 0.30–0.44 m, unos 0.33 m² en 360° (0.16 m² en
>     el semicírculo frontal).
>   - Tapa a z = 0.12 m: r ≈ 0.33–0.44 m.
>   - Agarre solo (sin mirar con la cámara): r ≈ 0.10–0.44 m. El rango mínimo de la cámara quita el
>     disco r < 0.30 m.
> - **Carga útil — EXCEDIDA.** El Lite 6 carga 600 g. La herramienta suma:
>   - Astra Pro Plus: 310 g.
>   - Vacuum Gripper Lite: 250 g.
>   - Soporte (131 cm³ de PLA): 105 g con ~50 % de relleno, 163 g macizo.
>   - 4 M4 con insertos: 14 g.
>   - Total: **679–737 g**, con CoG en (0, -42, 25) mm de `link_eef`. Antes de levantar cualquier
>     objeto ya se pasa por 79–137 g. Las masas de la sim ya son estas (`camera_body` 0.31 kg,
>     `other_geometry` 0.37 kg). En el robot real hay que configurar la carga en `set_tcp_load`.
>     Hay que decidir una de estas opciones:
>     - Cámara más liviana.
>     - Sacar la carcasa de la Astra.
>     - Cámara fija (eye-to-hand).
>     - Aceptar la sobrecarga, con menos velocidad y aceleración, y verificar con UFactory.
> - `grasp_test.py` (vista cenital a 0.25 m) solo sirve con `camera_model:=generic` y
>   `cloud_min_range:=0`.

---

## Fase 3 — Cámara real sin robot (cuando esté la cámara)

1. Driver solo:
   - ZED: `zed_camera.launch.py camera_model:=zed2i publish_tf:=false`.
   - Orbbec: `time_domain:=system`.
2. `charuco_detector` con el preset real. Mover el tablero de aluminio a mano y verificar:
   - Reproyección < 0.5 px en `~/debug_image`.
   - Detección estable a 0.25–0.40 m e inclinación 45°.
   - Reflejos del metal: probar `--invert`, luz difusa y ángulos.
3. Exactitud de distancia: comparar la `z` de `board_pose` con una medición de cinta a 0.30 m
   (< 2 mm). Si falla, recalibrar intrínsecos con `camera_calibration`.
4. `suction_grasp` con la cámara en trípode y un TF estático medido a mano `link_base → camera_link`.
   Verificar la segmentación de mesa y objetos en la nube real (ruido ZED vs Orbbec). Ajustar
   `table_threshold`, `voxel_size` y `cluster_radius`.

Aceptación: detector fiable sobre la placa grabada final; parámetros de `suction_grasp` afinados.

---

## Fase 4 — Taller

1. Grabar una prueba en un retazo de aluminio y fotografiarla con la cámara (Fase 3). Luego grabar la
   placa final.
2. Medir con el calibrador `square_length` y `marker_length` (promedio de varios cuadrados) y escribir
   los valores en `board.*` de `config/lite6_handeye.yaml`.
3. Sacar del CAD del soporte sándwich:
   - `x_prior_xyz_rpy` (`link_eef` → frame óptico).
   - Caja de colisión `tool_height/length/width`.
   - `suction_offset` (`link_eef` → punta de la ventosa comprimida).

---

## Fase 5 — Día del robot real (checklist)

1. Verificar la IP del Lite 6 (`robot_ip` por defecto es un placeholder).
2. `bringup.launch.py robot_ip:=<IP> camera:=zed`: comprobar que el TF `link_base → link_eef` coincide
   con la pose de la consola UFactory.
3. Primer movimiento: `velocity_scaling` 0.05, paro de emergencia a mano y una sola pose con
   `n_poses:=1`.
4. Calibración completa con el `poses_file` validado en la Fase 1 (puede requerir regenerarlo con
   `board_pose_base` real).
5. Criterio: `trans_rms_mm` < 2 y `rot_rms_deg` < 0.3. Si no se cumple, revisar la rigidez del soporte
   y las medidas del tablero.
6. Verificación independiente: tocar con la ventosa (sin vacío, a baja velocidad) una esquina conocida
   del tablero usando la pose calculada. El error esperado es < 3 mm.

---

## Fase 6 — Robot real con Orbbec: estado al 2026-11-16 y cómo mejorar la calibración

Estado actual (suficiente para seguir personas: 0.7° ≈ 2.5 cm a 2 m):
- **Intrínsecos RGB calibrados** (`docker/data/orbbec_intrinsics.yaml`, lo carga `orbbec_uvc`):
  fx 967.8 ± 9.3, fy 966.4, cx 632.4, cy 364.6, k1 0.114, k2 −0.198 (tangencial y k3 = 0),
  RMS 0.55 px, 20 vistas a 35–57 cm. Esquinas en `orbbec_intrinsics_corners.npz`.
  Se capturan con `docker/data/sim_scripts/calib_intrinsics.py` (grilla 3x3, < 60 cm).
  Lección: con el tablero a 1–2 m la focal queda ±5% (fx y distorsión no se separan).
- **Hand-eye** (`lite6_zed_calib.yaml`, 11 muestras, con K nominal HFOV 66.1°):
  `link_eef → óptico` t = (11.3, −69.2, 38.1) mm, yaw 180°. σ ≈ 1.5 mm / ~1° (submuestreo).
- **CAD** (`~/Work/adaptador_orbbec`, STEP de la Astra): lente RGB en x = +12.5 mm del cuerpo,
  centro y = −70 mm, vidrio a z = 43.3–44.3 mm del flange → `x_prior` (12.5, −70, 42, 0, 0, π).
  Diferencia con el hand-eye: (−1.2, +0.8, −3.9) mm y 0.7°; los ~4 mm en z son el centro óptico
  detrás del vidrio (el CAD no lo modela).

Para una calibración mejor (tareas de precisión: agarre, medición):
1. Repetir el hand-eye con los intrínsecos calibrados (hoy se resolvió con la K nominal).
2. Pose inicial con la cámara mirando hacia abajo al tablero a ~30 cm y **J4, J6 ≈ 0**
   (p. ej. `[-15, 4, 63, 1, 59, -5]°` para el tablero en (0.27, 0, 0)): con `wrist_window_deg: 170`
   la muñeca cubre casi todo su rango. En la corrida anterior 7/20 poses fallaron por la ventana
   (J6 inicial 93°), no por alcance.
3. Ya configurado: `n_poses: 25`, `roll_max_deg: 45`, `wrist_window_deg: 170`.
4. Criterio de diversidad (`sim_scripts/analyze_calib.py`): autovalores de los ejes de rotación
   lo más parejos posible (la corrida anterior: 0.05 / 0.13 / 0.82 = casi un solo eje).
5. Validar con `sim_scripts/resolve_calib.py` (park/refine/horaud deben coincidir a ~1 mm) y
   contra el CAD (diferencia esperada < 5 mm, < 1°).
6. Opcional: intrínsecos de fábrica (RGB e IR) desde el sensor de profundidad `2bc5:060f` con el
   SDK OpenNI2 de Orbbec (propiedad `CAM_PARAMS`), solo como verificación; suelen venir para
   640x480 (escalar a 1280x720). La red bloquea GitHub release-assets: bajar el SDK por otra vía.

Seguimiento de personas — re-identificación (2026-11-16):
- `byte_tracker.py` guarda un descriptor de ropa por track (histogramas HSV de torso y piernas).
  Los IDs que expiran (`track_buffer_s`) pasan a una galería por `reid_buffer_s` (120 s). Un
  track nuevo, en sus primeros 15 cuadros, hereda el ID de la galería o de un track perdido si la
  similitud es >= `reid_thresh` (0.8) con margen 0.05 sobre el segundo candidato.
- El objetivo elegido no se suelta mientras su ID esté en la galería; la app web muestra
  "en memoria". Pendiente: medir la similitud real con personas en el taller y ajustar
  `reid_thresh`. Si la ropa no basta (gente vestida parecido), cambiar el descriptor por un
  modelo ReID ONNX (p. ej. OSNet) en `appearance_feature`.

Seguimiento de personas — vibración y afinado (2026-11-16):
- Causas encontradas: pose de la cámara atrasada (/joint_states a 10 Hz; ahora se usa
  /ufactory/joint_states a 150 Hz e interpolación al instante de la imagen), overrides `-p` de
  pt_real.sh ignorados (el YAML gana; ahora van en un 2.º params file) y resonancia de muñeca a 12 Hz
  (J4/J6) con la carga de herramienta en 0 kg en el controlador.
- **Estado 2026-11-16:** cuna nueva (disco 10 mm) + ventosa + cámara montadas; carga `set_tcp_load(0.658, [0, -40, 28])`
  guardada con `save_conf()`; seguimiento fluido validado en el robot.
- Historial: carga anterior `set_tcp_load(0.653, [0, -41, 26])` (cuna 5 mm, 653 g con cámara 310 g,
  ventosa 250 g, PLA al 15 %); con la cuna nueva de 10 mm: `0.658, [0, -40, 28]`. Sin `save_conf()` se pierde al
  reiniciar el controlador. Leer `a.tcp_load` del SDK ~2 s después de conectar (antes muestra 0). Pesar la herramienta.
- Afinado: fase A sin robot (`tuning_record_host.sh` + `tuning_replay.py`, grabación `diag/tuning1`) y
  fase B en el robot (`tuning_live_host.sh`, modo `--ab`). Valores elegidos en `config/lite6_handeye.yaml`:
  cam_radius_gain 0, cam_z_gain 0, kf_accel_std 0.1, kf_meas_depth_std 0.6, max_joint_acc 0.8,
  joint_deadband 0.03, lookahead 0.10, image_latency 0.03. Resultado en el robot: vibración TCP
  1.56 ± 0.05 mm (antes 2.36 mm). La variación entre corridas es ~±40 %: comparar siempre A/B.
- **Causa principal de la falta de fluidez (resuelta 2026-11-16):** `lite6_traj_controller` tenía
  `interpolate_from_desired_state: false`: cada tramo de 20 ms del tracker arrancaba desde la posición real
  (atrasada ~600 ms) y el controlador pedía velocidades de recuperación (J1 3.4x la referencia, J4/J6
  vibrando a ~14 Hz, cientos de inversiones). Con `true` el seguimiento quedó fluido. Es de solo lectura:
  `sim_scripts/jtc_streaming.sh` recarga el controlador con `docker/data/jtc_streaming.yaml`;
  `run_real_bringup.sh` lo llama solo. Si se lanza `bringup.launch.py` a mano, correr `jtc_streaming.sh` después.
- `max_joint_vel_rad_s` 0.2 dejaba al brazo 1.9 s atrás de la persona: usar 0.5 (por defecto en pt_real.sh).
- Sigue alto el error de apuntado medio (~14°: zona muerta 5° + retraso) y los arranques
  (~17/min): siguiente mejora sería reemplazar la zona muerta por un control continuo de baja ganancia.

---

## Fuera de alcance / decisiones pendientes

- **Parte 1 del plan original (agente de programación autónomo):** no implementada. Es un proyecto
  independiente y hay que decidir si se hace y dónde.
- **Ejecución del pick con ventosa** (planificar `pregrasp → grasp`, activar el vacío y levantar):
  `suction_grasp` hoy solo publica las poses. Requiere el control del vacío (`xarm_api`
  `set_vacuum_gripper` o salida digital) y una máquina de estados del pick.
- Repo git: `git@github.com:Lich00o/Ensamblador_lite6.git` (rama `main`, desde 2026-11-16).
