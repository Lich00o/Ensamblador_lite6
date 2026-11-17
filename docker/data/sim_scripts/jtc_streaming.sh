#!/bin/bash
# Recarga lite6_traj_controller con interpolate_from_desired_state: true (jtc_streaming.yaml).
# El person_tracker manda la referencia en tramos de 20 ms; con el valor de fábrica (false) cada tramo
# arranca desde la posición real (atrasada) y el controlador pide velocidades de recuperación:
# J1 3.4x la referencia, J4/J6 vibrando a ~14 Hz. El parámetro es de solo lectura: hay que recargar.
# El brazo queda quieto durante la recarga. Idempotente. Uso (en lite6_gpu): jtc_streaming.sh
source /ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
YAML=/home/ubuntu/.ros/lite6_handeye/jtc_streaming.yaml
C=lite6_traj_controller
if ros2 param get /$C interpolate_from_desired_state 2>/dev/null | grep -q True; then
  echo "$C: interpolate_from_desired_state ya está en True"; exit 0
fi
TRK=$(ros2 node list 2>/dev/null | grep -c "^/person_tracker$")
[ "$TRK" -gt 0 ] && ros2 service call /person_tracker/set_tracking std_srvs/srv/SetBool "{data: false}" >/dev/null && sleep 1
call() { ros2 service call /controller_manager/$1 controller_manager_msgs/srv/$2 "$3" | grep -q "ok=True" || { echo "falló $1"; exit 1; }; }
call switch_controller SwitchController "{deactivate_controllers: [$C], strictness: 2}"
call unload_controller UnloadController "{name: $C}"
ros2 param set /controller_manager $C.params_file "[\"$YAML\"]" >/dev/null
call load_controller LoadController "{name: $C}"
call configure_controller ConfigureController "{name: $C}"
call switch_controller SwitchController "{activate_controllers: [$C], strictness: 2}"
[ "$TRK" -gt 0 ] && ros2 service call /person_tracker/set_tracking std_srvs/srv/SetBool "{data: true}" >/dev/null
echo "$C: interpolate_from_desired_state = $(ros2 param get /$C interpolate_from_desired_state | grep -o 'True\|False')"
