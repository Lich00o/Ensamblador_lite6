# Entorno Docker de simulación

Ubuntu 24.04 + ROS 2 Jazzy (`osrf/ros:jazzy-desktop-full`), con:

- MoveIt 2
- Gazebo Harmonic + `ros_gz` + `gz_ros2_control`
- `xarm_ros2` (rama `jazzy`) y `easy_handeye2`, compilados en `/ws`
- `lite6_handeye`

Sirve para trabajar desde un host que no es Ubuntu (Arch/Omarchy).

Todos los comandos se ejecutan desde la raíz del repo.

## Build

```bash
docker compose -f docker/compose.yaml build sim
```

La primera vez tarda unos 15–30 min y la imagen pesa unos 8 GB. Las capas de `xarm_ros2` quedan en
caché: solo se reconstruyen si cambia `docker/sim.repos`.

## Ejecutar

```bash
docker compose -f docker/compose.yaml run --rm sim           # GPU integrada AMD (/dev/dri)
docker compose -f docker/compose.yaml run --rm sim-nvidia    # RTX 3050 (PRIME offload)
```

Para abrir otra terminal en el mismo contenedor:

```bash
docker exec -it $(docker ps -qf ancestor=lite6_zed:jazzy | head -1) bash
```

Volúmenes:

| Host | Contenedor | Uso |
| --- | --- | --- |
| `src/lite6_handeye` | `/ws/src/lite6_handeye` | Se edita desde el host. Con `--symlink-install`, los cambios en Python se ven sin recompilar. |
| `boards/` | `/ws/boards` | Solo lectura. |
| `docker/data/` | `~/.ros/lite6_handeye` | Resultados de calibración y poses. |

Si agregas un ejecutable, un launch o un config nuevo, recompila dentro del contenedor:

```bash
colcon build --symlink-install --packages-select lite6_handeye
```

## Simulador del controlador UFactory (`uf-sim`)

```bash
docker compose -f docker/compose.yaml up -d uf-sim      # firmware Lite 6 + UFACTORY Studio
# Studio: http://127.0.0.1:18333    driver: bringup.launch.py robot_ip:=127.0.0.1
docker compose -f docker/compose.yaml stop uf-sim
```

Imagen `danielwang123321/uf-ubuntu-docker:latest` (firmware v2.4.0), enlazada desde la documentación
de UFactory. Usa los puertos 502–504, 18333 y 30000–30003 del host.

| Funciona | No funciona |
| --- | --- |
| Conexión TCP del driver | **Modo servo (modo 1)**: no ejecuta `set_servo_angle_j` |
| Servicios `xarm_api` en modo 0 (`set_servo_angle`) | Trayectorias de MoveIt/ros2_control |
| UFactory Studio | |

Para MoveIt usa `fake:=true`.

## Verificación (Fase 0)

```bash
colcon test --packages-select lite6_handeye --event-handlers console_direct+
colcon test-result --verbose
python3 -m pyflakes src/lite6_handeye
glxinfo -B | grep -E "renderer|version"     # aceleración GPU
rviz2                                       # GUI
```

## Notas

- GUI: en Omarchy/Hyprland, XWayland acepta conexiones locales y no hace falta `xhost`. Si aparece
  `cannot open display`, instala `xorg-xhost` y ejecuta `xhost +SI:localuser:$USER`.
- Verificado (2026-11-09):
  - `sim`: AMD Radeon (radeonsi, Mesa 25.2).
  - `sim-nvidia`: RTX 3050 (driver 610.57).
  - En ambos, OpenGL 4.6 y RViz funcionan.
- **Sin rosdep:** la base de datos de rosdep vive en `raw.githubusercontent.com`, que esta red bloquea
  (Fortiguard SDNS). `install_deps.py` resuelve las dependencias de los `package.xml` directamente con
  apt. Si un paquete nuevo trae una clave que no es `ros-jazzy-*` ni `python3-*`, agrégala a `MANUAL`.
- `ROS_DOMAIN_ID=42` y `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` aíslan el contenedor de otros nodos
  ROS de la red. Para hablar con el ROS del host, usa el mismo `ROS_DOMAIN_ID` en ambos.
- `VIDEO_GID` y `RENDER_GID` (por defecto 983 y 987) deben coincidir con
  `stat -c %g /dev/dri/card0 /dev/dri/renderD128` del host.
