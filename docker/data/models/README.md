# Modelos de detección

`yolo26n.onnx`: Ultralytics YOLO26n (COCO, 80 clases), exportado de `yolo26n.pt`
(HuggingFace `Ultralytics/YOLO26`) con ultralytics 8.4.215:

    YOLO('yolo26n.pt').export(format='onnx', imgsz=640, opset=12, simplify=True, end2end=False)

Salida cruda `(1, 84, 8400)` (cx, cy, w, h + 80 puntajes); el NMS lo hace `PersonDetector`.
Se ejecuta con onnxruntime: OpenCV 4.6 (apt de Ubuntu 24.04) no importa el bloque de
atención de YOLO26. GitHub release-assets está bloqueado por la red; usar HuggingFace.

`yolo26n-pose.onnx`: Ultralytics YOLO26n-pose (17 puntos COCO), mismo origen y exportación con
`yolo26n-pose.pt`. Es el que usa `pt_real.sh`: apunta a la cabeza con los puntos de la cara.

`osnet_ain_x1_0_msmt17.onnx`: re-identificación de personas (OSNet-AIN x1.0, MSMT17; licencia MIT),
de HuggingFace `mprib/osnet-ain-x1-0-msmt17-onnx`. Entrada `(N, 3, 256, 128)` RGB normalizada
ImageNet, salida `(N, 512)`. SHA-256 `f11a8fa82c3980f13f563210ddf197346bbd8d1bd0dca364c4ad6c4e39794116`.

    curl -L -o osnet_ain_x1_0_msmt17.onnx \
      https://huggingface.co/mprib/osnet-ain-x1-0-msmt17-onnx/resolve/main/osnet_ain_x1_0_msmt17.onnx

Los `.onnx` no se versionan (`.gitignore`): descargarlos o exportarlos en `docker/data/models/`.
