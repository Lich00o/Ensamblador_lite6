# Ensamblador Lite 6

Un brazo UFactory Lite 6 con una cámara en la muñeca **reconoce las piezas que le muestras y las
monta en la mesa de trabajo siguiendo un ensamble en STEP**.

```
 Persona muestra una pieza ──► el robot se orienta hacia ella y la reconoce
                                   │
          la pieza se deja en la zona de entrega
                                   │
 Ensamble .STEP ──► plan: orden de montaje + cara de agarre + pose destino de cada pieza
                                   │
                  pick & place con ventosa ──► el ensamble queda armado en la mesa
```

Estado: **la base de visión y control funciona en el robot real**: calibración cámara-robot,
seguimiento de personas y adaptador de herramienta. El montaje desde STEP es la etapa en curso. Ver
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Módulos

| Módulo | Qué hace | Estado |
|---|---|---|
| Calibración hand-eye (`auto_calibrator`, `charuco_detector`) | Ubica la cámara respecto de la brida recorriendo poses frente a un ChArUco | ✅ Robot real, 1.9 mm |
| Seguimiento de personas (`person_tracker` + app web) | Orienta la cámara hacia la persona elegida (YOLO pose, ByteTrack, re-ID OSNet, Kalman, Ruckig) | ✅ Robot real, fluido |
| Agarre con ventosa (`suction_grasp`) | Calcula el punto y la orientación de agarre desde la nube de puntos | 🟡 Validado en Gazebo; falta ejecutar el pick |
| Adaptador cámara + ventosa (`cad/adaptador_orbbec`) | Soporte paramétrico impreso | ✅ v3 montado, carga configurada |
| Mesa de trabajo calibrada, pose de piezas | Marco `workbench`, zonas, pose (x, y, giro) de cada pieza | ⏳ Etapas 2–3 |
| "Muéstrale y enséñale" | Reconoce la pieza mostrada y la lleva a su lugar | ⏳ Etapa 4 |
| STEP → plan de montaje | Orden, agarre y pose destino desde el ensamble | ⏳ Etapa 5 |

## Hardware

- UFactory Lite 6, firmware v2.8.2, en modo servo vía `ros2_control`.
- Orbbec Astra Pro Plus: RGB 1280 × 720 por UVC, con profundidad.
- Vacuum Gripper Lite (UFactory).
- PC con GPU NVIDIA (RTX 3050), con la detección y el re-ID en ONNX Runtime CUDA.

## Estructura

```
src/lite6_handeye/     paquete ROS 2 Jazzy (ament_python): nodos, launch, config, simulación
docker/                imagen Ubuntu 24.04 + Jazzy + GPU, compose, puente de cámara
docker/data/           montado en el contenedor como ~/.ros/lite6_handeye/
  sim_scripts/         lanzadores del robot real (run_real_bringup.sh, pt_real.sh), app web,
                       herramientas de diagnóstico y afinado
  models/README.md     de dónde bajar los modelos ONNX (no se versionan)
  *.yaml               calibraciones (hand-eye, intrínsecos), config del controlador
cad/adaptador_orbbec/  soporte cámara + ventosa (FreeCAD paramétrico, STEP/STL)
boards/                tablero ChArUco para grabar en láser
docs/                  documentación (ver abajo)
```

## Puesta en marcha rápida (robot real)

```bash
docker/camera_bridge.sh                                                    # host: cámara -> contenedor
docker start lite6_gpu
docker exec lite6_gpu bash /home/ubuntu/.ros/lite6_handeye/sim_scripts/run_real_bringup.sh
docker exec -it lite6_gpu bash /home/ubuntu/.ros/lite6_handeye/sim_scripts/pt_real.sh true   # mueve el brazo
# app web: http://localhost:8080
```

Guardar el robot en su caja (sin herramientas; J1 = 180°, resto 0°):

```bash
docker exec lite6_gpu python3 /home/ubuntu/.ros/lite6_handeye/sim_scripts/pose_guardado.py
```

Antes de esto: crear el entorno Docker (`docker/README.md`), bajar los modelos (`docker/data/models/README.md`)
y calibrar (`docs/CALIBRACION_HANDEYE.md`).

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Objetivo final, supuestos y etapas con criterios de aceptación |
| [`docs/ARQUITECTURA_SEGUIMIENTO.md`](docs/ARQUITECTURA_SEGUIMIENTO.md) | Arquitectura del lazo de seguimiento, parámetros, cómo lanzar la app y problemas conocidos |
| [`docs/CALIBRACION_HANDEYE.md`](docs/CALIBRACION_HANDEYE.md) | Paquete, tablero, intrínsecos, simulación y calibración hand-eye |
| [`docs/PLAN_PENDIENTE.md`](docs/PLAN_PENDIENTE.md) | Bitácora técnica: fases de validación, mediciones y causas de los problemas resueltos |
| [`docker/README.md`](docker/README.md) | Entorno Docker y simulación |
| [`cad/adaptador_orbbec/README.md`](cad/adaptador_orbbec/README.md) | Adaptador: geometría, impresión y carga |

## Seguridad

- La app web y los scripts **mueven el robot**. Ten siempre la parada de emergencia al alcance.
- La app web no tiene autenticación y escucha solo en `127.0.0.1`.
- La herramienta completa (0.658 kg) supera la carga nominal del Lite 6 (0.6 kg). Usa velocidades moderadas.

## Contenido de terceros no incluido

No se incluyen, por licencia:
- los modelos ONNX (YOLO26 de Ultralytics y OSNet-AIN);
- los STEP de la cámara Orbbec y de la ventosa UFactory;
- las mallas del maniquí de Gazebo.

Cada README indica de dónde obtenerlos.
