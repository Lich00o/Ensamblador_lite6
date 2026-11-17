#!/usr/bin/env python3
"""Lleva el Lite 6 a la pose de guardado para su caja de fábrica: J1 = 180°, resto 0° (brazo plegado
mirando al lado opuesto; con J1 = 0 choca con la caja).

SOLO SIN HERRAMIENTAS: pone el modelo de colisión de herramienta en "ninguna" (con el modelo de pinza
el controlador da C22 al girar plegado) y la carga del TCP en 0 kg. Ninguno de los dos se guarda con
save_conf(): al reiniciar el controlador vuelven los valores guardados (carga 0.658 kg de la cuna).
Detén antes el tracker y el bringup. Ten la parada de emergencia a mano.

Uso (host):
  docker exec lite6_gpu python3 /home/ubuntu/.ros/lite6_handeye/sim_scripts/pose_guardado.py [j1]
  j1 = 180 por defecto; -180 llega a la misma postura girando al revés.
"""
import sys
import time

from xarm.wrapper import XArmAPI

IP = '192.168.1.153'
SPEED = 10  # °/s

j1 = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
a = XArmAPI(IP)
time.sleep(2)
print('modelo herramienta -> ninguna:', a.set_collision_tool_model(0))
a.clean_warn()
a.clean_error()
time.sleep(0.5)
a.motion_enable(True)
a.set_mode(0)
a.set_state(0)
time.sleep(1)
a.set_tcp_load(0, [0, 0, 0])
print('antes:', [round(x, 1) for x in a.angles[:6]], '| error', a.error_code, 'state', a.state)
if a.error_code == 0 and a.state in (0, 1, 2):
    print('mover:', a.set_servo_angle(angle=[j1, 0, 0, 0, 0, 0], speed=SPEED, wait=True))
    print('final:', [round(x, 1) for x in a.angles[:6]], '| error', a.error_code)
else:
    print('NO se movió: sigue en error')
a.disconnect()
