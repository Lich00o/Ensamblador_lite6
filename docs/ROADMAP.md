# Hoja de ruta — Ensamblador Lite 6

## Objetivo final

1. El robot se orienta hacia la persona, que le **muestra una pieza**.
2. El robot **reconoce** la pieza y sabe **dónde va** en la mesa de trabajo.
3. Con un **ensamble en STEP**, el robot **apila o monta las piezas** en la mesa con la misma
   disposición que tienen en el ensamble.

## Supuestos de diseño

- **El robot está fijo y alcanza ~44 cm.** "Acercarse" significa girar y estirarse hacia la persona dentro de ese alcance. La mesa de trabajo tiene que estar dentro del alcance útil (radio ≈ 0.30–0.44 m, según `observable_map.py`).
- **La pieza no se toma de la mano de la persona.** Se muestra a la cámara y luego se deja en una **zona de entrega** de la mesa, desde donde el robot la toma con la ventosa. Es más robusto y más seguro.
- **Restricciones de las piezas:**
  - livianas: la herramienta ya pesa 0.658 kg, por encima de la carga nominal de 0.6 kg;
  - con al menos una cara plana para la ventosa;
  - tolerancias de montaje de algunos milímetros: la calibración hand-eye da ~2 mm y la profundidad de la Astra es ruidosa.
- **Se empieza con bloques simples:** cubos y prismas impresos de 20–50 g, y un ensamble STEP de 3–4 bloques apilados. Las piezas reales vienen después.

## Etapas

Cada etapa produce algo usable por sí solo.

### 0. Base (hecho)

- Calibración hand-eye automática (`auto_calibrator`), con 1.9 mm en el robot real.
- Seguimiento de personas fluido, con IDs y re-ID (`person_tracker`, `docs/ARQUITECTURA_SEGUIMIENTO.md`).
- Puntos de agarre con ventosa desde la nube de puntos (`suction_grasp`), validado en Gazebo con menos de 0.3 mm.
- Adaptador de la cámara y la ventosa (`cad/adaptador_orbbec`), con su carga configurada en el robot.

### 1. Pick & place con ventosa

- Control del vacío: hay que confirmar si va por la salida digital de la herramienta o por `set_vacuum_gripper` del driver.
- Máquina de estados: aproximación → contacto → vacío → verificación de agarre → levantar → transporte → soltar.
- Planificación con MoveIt, con la caja de colisión real de la herramienta.
- **Aceptación:** 20 de 20 bloques movidos de una zona A a una posición fija B, con error ≤ 3 mm.

### 2. Mesa de trabajo calibrada

- Un tablero ChArUco fijo sobre la mesa define el marco `workbench` (lo mide `charuco_detector`).
- Zonas: entrega, armado y descarte.
- **Aceptación:** dejar un bloque en coordenadas `workbench` dadas, con error ≤ 3 mm.

### 3. Pose de las piezas sobre la mesa

- Vista superior y 3 grados de libertad (x, y, giro): una pieza apoyada sobre una cara conocida no necesita pose 6D.
- Detección 2D + profundidad o contorno + ajuste del modelo (ICP en 2D sobre la silueta).
- La profundidad de la Astra empieza en ~0.6 m: se usan vistas oblicuas, como `pose_sampler.observation_viewpoints`.
- **Aceptación:** pose de la pieza con error ≤ 3 mm y ≤ 3°.

### 4. "Muéstrale y enséñale"

- Al mostrar la pieza, se toma el recorte junto a las muñecas detectadas por YOLO pose y se calcula un embedding visual (por ejemplo, DINOv2 en ONNX).
- Modo enseñanza: se muestra la pieza y se indica su lugar con un clic en la app web. Se guarda el par (embedding, pose destino).
- Modo trabajo: se muestra la pieza y se deja en la zona de entrega. El robot la reconoce, la toma y la deja en su lugar.
- **Aceptación:** con 4 piezas distintas, reconocerlas correctamente en ≥ 95 % de los casos.

### 5. STEP → plan de montaje

- Leer el ensamble con FreeCAD / OpenCascade: piezas, geometría y pose de cada pieza en el ensamble.
- Orden de montaje por contactos y gravedad: una pieza va antes que las que se apoyan sobre ella.
- Cara de agarre por pieza: plana, hacia arriba en la pose final, con área suficiente para la ventosa y cerca del centro de masa.
- Pose destino = pose en el ensamble llevada al marco `workbench`. El origen del ensamble se ancla en la zona de armado.
- Simulación previa en Gazebo con las piezas del STEP como modelos.
- **Aceptación:** apilar en el robot real un ensamble de 3–4 bloques con error ≤ 3 mm por pieza.

### 6. Reconocimiento desde el CAD

- Identificar la pieza mostrada comparándola con renders del STEP, sin enseñarla antes.
- Opcional: pose 6D desde el CAD (por ejemplo, FoundationPose), aunque es exigente para los 4 GB de la RTX 3050.
- Verificación visual después de cada colocación, con corrección o aviso.

## Preguntas abiertas

- ¿Cómo se activa el vacío del Vacuum Gripper Lite en esta instalación?
- Dimensiones y ubicación de la mesa de trabajo respecto de la base del robot.
- ¿Se aligera el adaptador o se cambia la cámara, para volver a quedar dentro de los 600 g?
- Tipo de piezas finales: material, tamaño y caras planas disponibles.
