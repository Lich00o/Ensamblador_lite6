# Seguimiento de personas con el Lite 6 — arquitectura y puesta en marcha

Este documento explica cómo funciona la app que hace que el Lite 6 siga a una persona con la cámara
Orbbec Astra Pro Plus montada en la muñeca, y cómo lanzarla en el robot real. La calibración
hand-eye está en `docs/CALIBRACION_HANDEYE.md`; lo pendiente y el historial de ajustes, en `docs/PLAN_PENDIENTE.md`.

---

## 1. Vista general

```
 HOST (Arch)                                   CONTENEDOR lite6_gpu (Ubuntu 24.04, ROS 2 Jazzy, RTX 3050)
 ──────────────────────────                    ─────────────────────────────────────────────────────────
 Orbbec /dev/video0 (UVC)                       orbbec_uvc ──► /camera/color/image_raw (+ camera_info)
   │ ffmpeg (camera_bridge.sh)                      │            stamp = hora de escritura del JPEG
   ▼                                                ▼
 docker/data/.live_cam.jpg  ═══ montado ═══►   person_tracker ──────────────► /lite6_traj_controller/
 (= ~/.ros/lite6_handeye/ en el contenedor)        │  ▲   │  YOLO26n-pose (ONNX, CUDA)    joint_trajectory (50 Hz)
                                                   │  │   │  ByteTrack + re-ID OSNet           │
 Navegador ◄── http://localhost:8080 ──── webapp   │  │   │  Kalman → pose 6-DOF → IK          ▼
   (video, IDs, seguimiento, jog, STOP)    (FastAPI)  │   │  suavizador Ruckig            lite6_traj_controller
                                                   │  │   ▼                              (ros2_control, 150 Hz)
                                                   │  └── /ufactory/joint_states (150 Hz)      │ set_servo_angle_j
                                                   ▼                                           ▼
                                        /person_tracker/{tracks,target_point,          Lite 6 (192.168.1.153)
                                         target_eef_pose,debug_image}                  modo servo (mode 1)
```

Todo lo de ROS corre en **un solo contenedor** (`lite6_gpu`). Si los nodos quedan repartidos entre
contenedores, el DDS pierde las imágenes y el `camera_info`.

## 2. Componentes

### 2.1 Host

| Pieza | Qué hace |
|---|---|
| `docker/camera_bridge.sh` | `ffmpeg` lee la Orbbec por UVC (MJPEG 1280×720 a 30 fps) y reescribe `docker/data/.live_cam.jpg` en cada cuadro. Tiene que estar corriendo para que haya imagen. Si la cámara se desconecta del USB, espera a que vuelva y relanza `ffmpeg`. |
| `docker/data/` | Montado en el contenedor como `~/.ros/lite6_handeye/`. Contiene la calibración, los modelos, los scripts (`sim_scripts/`) y los diagnósticos (`diag/`). |
| Navegador | Abre la app web en `http://localhost:8080`. El contenedor usa la red del host (`network_mode: host`). |

### 2.2 Contenedor `lite6_gpu`

Es un contenedor persistente creado a partir del servicio `sim-nvidia` de `docker/compose.yaml`,
con la imagen `lite6_zed:jazzy` y `runtime: nvidia`. La imagen ya trae `onnxruntime-gpu`, `ruckig`,
`fastapi` y `uvicorn`. Montajes:

- `src/lite6_handeye` → `/ws/src/lite6_handeye`: el paquete se compila con `--symlink-install`, así que los cambios en Python y en el YAML se aplican al reiniciar el nodo, sin recompilar;
- `docker/data` → `~/.ros/lite6_handeye`.

| Proceso | Lanzado por | Función |
|---|---|---|
| `bringup.launch.py` | `run_real_bringup.sh` | MoveIt + driver UFactory (`ros2_control`) + `orbbec_uvc` + TF de la calibración. El detector ChArUco solo arranca con `DETECTOR=true`, porque consume ~1 núcleo y solo sirve para calibrar |
| `person_tracker` | `pt_real.sh` | El seguimiento (sección 3) |
| `webapp/server.py` | `pt_real.sh` (si no hay otra instancia) | App web: video MJPEG, WebSocket de estado y comandos, jog |

### 2.3 Robot

- Lite 6 en `192.168.1.153`, firmware v2.8.2.
- El driver `ros2_control` lo pone en **modo servo** (mode 1) y le envía `set_servo_angle_j` a 150 Hz.
- La carga de la herramienta se configura en el controlador del robot (ver 2.4). Sin ella, el
  controlador compensa como si la muñeca estuviera vacía.

### 2.4 Herramienta: soporte, cámara y ventosa

El modelo 3D del soporte está en `~/Work/adaptador_orbbec`:

- `generar_soporte.py` es el generador paramétrico. Se ejecuta con `freecadcmd -c "exec(open('generar_soporte.py').read())"`.
- Genera `soporte_cuna.step/.stl` y `soporte_tapa.step/.stl`.
- El ensamble completo para revisión es `ensamble_orbbec_lite6_v3.solidcraft`.

| Versión | Disco del sándwich | Hueco del cable | Ventosa (cara de apoyo) | Masa total | Centro de gravedad (marco del flange) |
|---|---|---|---|---|---|
| Anterior (`respaldo_disco5mm/`) | 5 mm | 30 × 14 mm | z = 5 mm | 0.653 kg | (0, −41, 26) mm |
| **Nueva (v3)** | **10 mm**, crece solo hacia la ventosa | **33 × 17 mm** (pasa el filtro del cable) | **z = 10 mm** | **0.658 kg** | **(0, −40, 28) mm** |

Cómo se calculó la masa:

| Pieza | Masa |
|---|---|
| Cámara Astra Pro Plus (dato de tienda) | 310 g |
| Vacuum Gripper Lite | 250 g |
| Tornillos M4 + insertos | 14 g |
| Cuna nueva, laminada en Bambu Studio (0.20 mm, 2 paredes, 15 % de relleno), PLA / PETG | 73.4 g / 72.3 g |
| Tapa, mismo laminado | 11 g |

El centro de gravedad se calculó con el CAD. Conviene pesar la herramienta armada y corregir la masa.

La cámara no cambió de posición respecto del flange: la calibración hand-eye sigue valiendo con la
cuna nueva.

**Estado (2026-11-16):** la cuna nueva está montada con la ventosa y la cámara. La carga
`0.658 kg, [0, -40, 28]` está guardada en el robot con `save_conf()`.

**Al montar la cuna nueva:**

1. **Tornillos M6 de la ventosa:** 5 mm más largos que los anteriores.
2. **Carga en el robot.** Cambia la configuración del controlador del robot: ejecútalo con el robot quieto. `save_conf()` la guarda; sin esa llamada se pierde al reiniciar el controlador.
   ```bash
   docker exec lite6_gpu python3 -c "
   import time
   from xarm.wrapper import XArmAPI
   a = XArmAPI('192.168.1.153'); time.sleep(2)
   print('antes:', a.tcp_load)
   print('set_tcp_load:', a.set_tcp_load(0.658, [0, -40, 28]))
   print('save_conf:', a.save_conf())
   time.sleep(1); print('ahora:', a.tcp_load)
   a.disconnect()"
   ```
   Las dos llamadas deben devolver `0`, y `ahora:` debe mostrar `[0.658, [0.0, -40.0, 28.0]]`. Lee siempre `a.tcp_load` después de esperar ~2 s, porque antes del primer reporte muestra 0. Si vuelves a la cuna anterior, usa `0.653, [0, -41, 26]`.
3. **TCP de la ventosa.** Su punta queda 5 mm más lejos del flange. Si usas un TCP para la ventosa (en la consola de UFactory o con `set_tcp_offset`), súmale +5 mm en z. El seguimiento de personas no usa ese TCP: trabaja con `link_eef`.
4. **Simulación.** El `src/lite6_handeye/sim/meshes/soporte_camara.stl` de Gazebo sigue siendo la versión anterior y no incluye la ventosa.

## 3. El lazo de seguimiento (`person_tracker_node.py`)

Por cada imagen (~23–27 Hz):

1. **Pose de la cámara en el instante de la imagen.** Las articulaciones llegan de `/ufactory/joint_states` (150 Hz) y se guardan en un historial. Se interpolan al momento `stamp − image_latency_s` y se aplica la cinemática directa exacta (`lite6_fk`) junto con la calibración `link_eef → óptico`. No se usa `/joint_states`: va a 10 Hz y atrasaba la pose hasta 100 ms, lo que hacía oscilar el lazo.
2. **Detección.** YOLO26n-pose en ONNX sobre CUDA (`models/yolo26n-pose.onnx`). El punto al que se apunta es la cabeza, sacada de los puntos de la cara del modelo de pose.
3. **IDs por persona.** ByteTrack, con compensación del giro de la cámara: la rotación entre cuadros se conoce por cinemática y se aplica como homografía `K R K⁻¹` antes de asociar.
4. **Re-identificación.** Cada track guarda un embedding de OSNet-AIN x1.0 MSMT17 (`models/osnet_ain_x1_0_msmt17.onnx`).
   - Los IDs perdidos pasan a una galería que dura 120 s.
   - Una persona nueva hereda un ID de la galería si la similitud coseno es ≥ `reid_thresh` durante 3 cuadros seguidos, con margen sobre el segundo candidato.
   - El objetivo elegido no se suelta mientras su ID esté en la galería.
5. **Posición 3D.** Se traza un rayo desde el píxel de la cabeza y se usa la distancia estimada por el alto del bbox. Un filtro de Kalman 3D de velocidad constante en `link_base` confía poco en la profundidad (`kf_meas_depth_std`) y descarta saltos de más de N sigma.
6. **Pose deseada.** `compute_tracking_pose_6dof` orienta el eje óptico hacia la persona con el horizonte nivelado. Con `cam_radius_gain = cam_z_gain = 0` la cámara solo **gira**: no se traslada con la distancia ni con la altura estimadas, que son las señales más ruidosas.
7. **Zona muerta.** Si la persona está a menos de `deadband_enter_deg` del eje óptico, el brazo no se mueve. Al salir de esa zona recentra hasta quedar a menos de `deadband_exit_deg`.
8. **IK.** `lite6_ik_6dof` (Levenberg-Marquardt) se siembra con el último comando. Se descartan las soluciones que giran J4 o J6 más de `max_wrist_jump_deg` (cambio de rama de la muñeca). En ese caso se reintenta desde la pose de vigilancia girada en J1.
9. **Suavizador.** `JointSmoother` (Ruckig) genera, a 50 Hz, una referencia continua con límites de velocidad, aceleración y jerk.
10. **Envío.** Cada 20 ms se publica en `/lite6_traj_controller/joint_trajectory` un tramo corto: un punto a +`stream_lookahead_s` y un punto de frenado.

### Controlador de trayectoria: requisito para que el movimiento sea fluido

`lite6_traj_controller` **debe** tener `interpolate_from_desired_state: true`.

Con el valor de fábrica (`false`), cada tramo de 20 ms arranca desde la posición real, que va
~600 ms atrasada. El controlador pide entonces velocidades de recuperación: J1 llegaba a 3.4 veces
la referencia y J4/J6 vibraban a ~14 Hz.

El parámetro es de solo lectura, así que hay que recargar el controlador.
`sim_scripts/jtc_streaming.sh` lo hace con `docker/data/jtc_streaming.yaml`, y
`run_real_bringup.sh` lo llama automáticamente.

### Tópicos y servicios del tracker

| Nombre | Tipo | Uso |
|---|---|---|
| `~/debug_image` | `Image` | Video con cajas, IDs, objetivo y estado |
| `~/tracks` | `String` (JSON) | `{target, tracks:[{id, visible, score, bbox}], gallery:[ids]}` |
| `~/target_point` | `PointStamped` | Persona filtrada en `link_base` |
| `~/target_eef_pose` | `PoseStamped` | Pose deseada de `link_eef` |
| `~/select_target` | `Int32` (sub) | Elegir objetivo por ID (−2 = soltar) |
| `~/debug_image_mouse_left` | `Point` (sub) | Elegir objetivo por clic en la imagen |
| `~/set_tracking` | `SetBool` | Pausar o reanudar el seguimiento |
| `~/go_watch` | `Trigger` | Volver a la pose de vigilancia |

## 4. Parámetros

- **Base:** `src/lite6_handeye/config/lite6_handeye.yaml`, sección `person_tracker`. Contiene los valores afinados el 2026-11-16 con sus razones en los comentarios.
- **Overrides:** `pt_real.sh` fija los tópicos, la pose de vigilancia y los rangos de la cámara. Todo `-p param:=valor` extra que reciba se escribe en `/tmp/pt_overrides.yaml` y se pasa como **segundo** `--params-file`. En ROS 2 un `-p` suelto pierde contra la entrada `person_tracker:` del YAML, así que el segundo archivo es necesario para que el override gane.
- **En caliente:** casi todos los parámetros se leen en cada ciclo y se pueden cambiar con `ros2 param set /person_tracker <param> <valor>`. Los límites del suavizador (`max_joint_vel/acc/jerk`) y los del Kalman (`kf_*`) se aplican en caliente gracias a un callback de parámetros.

Valores clave del robot real:

| Parámetro | Valor | Nota |
|---|---|---|
| `max_joint_vel_rad_s` | 0.5 | Por defecto en `pt_real.sh`. Con 0.2 el brazo iba 1.9 s atrás |
| `max_joint_acc_rad_s2` / `max_joint_jerk_rad_s3` | 0.8 / 8 | Verificado A/B en el robot |
| `stream_lookahead_s` | 0.10 | 0.15 triplicó la vibración |
| `cam_radius_gain` / `cam_z_gain` | 0 / 0 | La cámara solo gira |
| `kf_accel_std` / `kf_meas_depth_std` | 0.1 / 0.6 | Afinado en la fase A |
| `deadband_enter_deg` / `deadband_exit_deg` | 5 / 2 | |
| `reid_thresh` | 0.8 | Con 0.6 fusionó a dos personas |

## 5. Cómo lanzar la app (robot real)

Requisitos: robot encendido y en red (`192.168.1.153`), Orbbec conectada al PC y el contenedor `lite6_gpu` creado.

**1. Puente de la cámara** (host; déjalo corriendo en su propia terminal):
```bash
cd ~/Projects/lite6_zed && docker/camera_bridge.sh
```

**2. Contenedor:**
```bash
docker start lite6_gpu
```

**3. Bringup** (driver, MoveIt, cámara ROS, TF calibrado y recarga del controlador). No mueve el brazo:
```bash
docker exec lite6_gpu bash /home/ubuntu/.ros/lite6_handeye/sim_scripts/run_real_bringup.sh
```
- Se lanza en una sesión propia (`setsid`): un `Ctrl+C` en otra terminal no lo mata.
- Su log queda en `/tmp/real_bringup.log` dentro del contenedor.
- Al final debe imprimir `JOINT_STATES OK`, `CAMERA RGB OK` y `interpolate_from_desired_state = True`.

**4. Tracker + app web.** El brazo se mueve: ten la parada de emergencia a mano.
```bash
docker exec -it lite6_gpu bash /home/ubuntu/.ros/lite6_handeye/sim_scripts/pt_real.sh true
```
- Primer argumento: `true` mueve el brazo; `false` es solo visión.
- Segundo argumento opcional: la velocidad máxima articular (por defecto 0.5).
- Después se pueden agregar overrides, por ejemplo `-p reid_thresh:=0.85`.
- Al arrancar va a la pose de vigilancia `[-5, -36, 25, -11, -29, -80]°`.
- `Ctrl+C` detiene el tracker y la app web que abrió este script.

**5. Usarla:** abre `http://localhost:8080`.
- Elige el objetivo con un clic sobre la persona o con su botón de ID, y activa **Seguimiento**.
- Si el robot aparece "NO LISTO" (por ejemplo, después de un parpadeo a state 4 o de un STOP), usa **Robot listo**: pone mode 1 / state 0 y reactiva el controlador. No limpia códigos de error; eso se hace desde la consola de UFactory.
- **STOP** pausa el seguimiento y detiene el robot (state 4). Para seguir, usa **Robot listo**.
- El jog solo se mueve mientras mantienes presionado el botón.
- La página no tiene autenticación y escucha solo en `127.0.0.1`. `WEB_HOST=0.0.0.0` la expone a la red: no lo hagas en una red compartida.

**Apagar:** `Ctrl+C` en el tracker y en el puente. El bringup se detiene con
`docker exec lite6_gpu pkill -f bringup.launch.py`, o con `docker stop lite6_gpu`.

## 6. Herramientas de diagnóstico y afinado (`docker/data/sim_scripts/`)

| Script | Uso |
|---|---|
| `rec_tracking.py <s> <nombre>` + `analyze_tracking.py diag/<nombre>.pkl` | Graba todo el lazo (articulaciones a 150 Hz, comandos, objetivo, pose deseada, tracks, estado del robot) y reporta ruido del objetivo, retardo del lazo, error de apuntado, seguimiento entre referencia y robot real, inversiones y oscilación por articulación |
| `tuning_record_host.sh` + `tuning_replay.py` | Fase A: graba ~2 min sin mover el robot (con instrucciones por voz) y repite la grabación con un robot virtual para barrer parámetros de percepción y filtro (`--sweep`, `--grid`, `--set`) |
| `tuning_live_host.sh [--ab k=v ...] <nombre>` | Fase B: recorre combinaciones de parámetros de movimiento en el robot. Usa siempre `--ab`: las corridas en el robot varían ~±40 % entre sí |
| `reid_live_test.py [--dump dir]` | Prueba del re-ID con la cámara en vivo, sin robot (video en `:8090`) |
| `check_watch_pose.py` | Verifica que una pose de vigilancia mantenga J5 lejos de 0 (singularidad de la muñeca) |
| `jtc_streaming.sh` | Recarga el controlador con `interpolate_from_desired_state: true` (idempotente) |

## 7. Problemas conocidos

| Síntoma | Causa / solución |
|---|---|
| Sin imagen | `camera_bridge.sh` no está corriendo, o `/dev/video0` está ocupado por otro programa |
| La cámara se cae seguido (`USB disconnect` en `journalctl -k`) | Cable o hub USB: conectar a un puerto directo del PC, usar un cable corto y de buena calidad, y sujetar el cable a lo largo del brazo para que no tire al moverse |
| Seguimiento lento y `person_tracker` a más de 150 % de CPU | Revisar el log del tracker al arrancar: `Detector:` y `ReID:` deben decir `CUDAExecutionProvider` |
| El brazo no se mueve y el video dice "SEGUIMIENTO PAUSA" o "Sin objetivo" | Elegir ID y activar seguimiento en la app web |
| El robot no recibe comandos después de un error | El driver dejó `lite6_traj_controller` inactivo: usar **Robot listo** |
| Movimiento a tirones, J4/J6 vibrando | El controlador perdió `interpolate_from_desired_state` (por ejemplo, se lanzó `bringup.launch.py` a mano): correr `jtc_streaming.sh` |
| El brazo va muy atrás de la persona | `max_joint_vel_rad_s` bajo (no usar 0.2) |
| Un `-p` no tiene efecto | Lanzar a través de `pt_real.sh`, que los pasa como segundo archivo de parámetros |
| Dos personas comparten ID | Subir `reid_thresh` (0.85–0.9) |
| `tcp_load` aparece en 0 en el SDK | Esperar ~2 s después de conectar antes de leer `a.tcp_load`: antes del primer reporte siempre muestra 0 |
| Error C19 (End Module Communication Error) | Revisar el cable y el conector de la herramienta. Limpiar el error en la consola de UFactory y después usar **Robot listo** |
