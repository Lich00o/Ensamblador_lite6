#!/bin/bash
# person_tracker en robot real con Orbbec + app web (video, seguimiento, objetivo, jog). Uso: pt_real.sh <enable_motion true|false> <max_joint_vel> [-p param:=valor ...]
# Ctrl+C detiene el tracker y la app web que abrió este script.
# Pose de vigilancia: revisar candidatas con check_watch_pose.py (J5 lejos de 0 = sin singularidad de
# muñeca). La pedida [0,-80,10,0,0,-90] tiene J5=0; se usa la más cercana con la cámara a la misma altura.
source /ws/install/setup.bash
export ROS_DOMAIN_ID=42
DIR="$(cd "$(dirname "$0")" && pwd)"

pids=()
# App web (video + seguimiento + objetivo + jog): http://localhost:8080 (WEB_HOST=0.0.0.0 la expone a la red,
# sin autenticación). El panel Tk (tracker_panel.py) sigue disponible por separado.
if ! pgrep -f "webapp/server.py|python3 server.py" >/dev/null; then
  (cd "$DIR/webapp" && python3 server.py --host "${WEB_HOST:-127.0.0.1}" --port "${WEB_PORT:-8080}" \
    > /tmp/web.log 2>&1) &
  pids+=($!)
fi
echo "App web: http://localhost:${WEB_PORT:-8080}"
cleanup() {
  for p in "${pids[@]}"; do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
}
trap cleanup EXIT

# Los -p de la línea de comandos son comodín (/**) y en ROS 2 pierden contra la entrada
# "person_tracker:" del YAML: todos los overrides van a un 2.º archivo de parámetros, que sí gana.
ARGS=(
  -p image_topic:=/camera/color/image_raw -p camera_info_topic:=/camera/color/camera_info
  -p joint_states_topic:=/ufactory/joint_states
  -p optical_frame:=camera_color_optical_frame
  -p onnx_model:=/home/ubuntu/.ros/lite6_handeye/models/yolo26n-pose.onnx
  -p "watch_joints_deg:=[-5.0, -36.0, 25.0, -11.0, -29.0, -80.0]" -p cam_radius_nom:=0.22 -p cam_z_nom:=0.30
  -p "cam_radius_range_m:=[0.18, 0.28]" -p "cam_z_range_m:=[0.22, 0.40]"
  -p enable_motion:=${1:-false} -p move_to_watch_on_start:=${1:-false}
  -p max_joint_vel_rad_s:=${2:-0.5} "${@:3}"
)
OVR=/tmp/pt_overrides.yaml
python3 - "$OVR" "${ARGS[@]}" <<'PY'
import sys, yaml
out, a, prm = sys.argv[1], sys.argv[2:], {}
for i, x in enumerate(a):
    if x == '-p':
        k, v = a[i + 1].split(':=', 1)
        v = yaml.safe_load(v)
        # 0 o [1, 2] llegarían como enteros y el parámetro es double: ROS 2 los rechaza
        f = lambda x: float(x) if isinstance(x, int) and not isinstance(x, bool) else x  # noqa: E731
        prm[k] = v if k in ('kf_max_rejects',) else ([f(x) for x in v] if isinstance(v, list) else f(v))
yaml.safe_dump({'person_tracker': {'ros__parameters': prm}}, open(out, 'w'))
print('overrides:', ', '.join(f'{k}={v}' for k, v in prm.items()))
PY

ros2 run lite6_handeye person_tracker --ros-args \
  --params-file /ws/install/lite6_handeye/share/lite6_handeye/config/lite6_handeye.yaml \
  --params-file "$OVR" -r __node:=person_tracker
