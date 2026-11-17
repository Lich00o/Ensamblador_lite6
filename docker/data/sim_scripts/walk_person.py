"""Animación humanoide articulada de 16 GDL a 50 Hz en Gazebo (caminar, sentadilla y saltar).

Publica comandos continuos `std_msgs/msg/Float64` a las 16 articulaciones funcionales de
`person_target`:
  - Base 4-DOF: `px`, `py`, `pz`, `yaw`
  - Tronco y cabeza: `pitch` (columna), `head_yaw` (cuello/cabeza)
  - Brazos (hombro + codo): `arm_l`, `elbow_l`, `arm_r`, `elbow_r`
  - Piernas (cadera + rodilla + tobillo): `leg_l`, `knee_l`, `ankle_l`, `leg_r`, `knee_r`, `ankle_r`
interpolados a 1000 Hz por Gazebo Harmonic (`gz-sim-joint-position-controller-system`).
"""

import math
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

JOINTS = (
    'px', 'py', 'pz',
    'yaw', 'pitch', 'head_yaw',
    'arm_l', 'elbow_l',
    'arm_r', 'elbow_r',
    'leg_l', 'knee_l', 'ankle_l',
    'leg_r', 'knee_r', 'ankle_r',
)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def choreography(t: float) -> tuple[dict[str, float], str]:
    """Devuelve el estado articular de 16 GDL con cinemática funcional de extremidades y fase.

    Convención de signos en `person_target` (mirando hacia el robot en -X, eje articular +Y):
      - `pitch < 0` inclina el torso hacia adelante (-X).
      - `head_yaw` rota la cabeza alrededor de +Z.
      - `arm > 0` lleva el brazo hacia adelante (-X) y arriba (+Z).
      - `elbow > 0` flexiona el antebrazo y la mano hacia adelante/arriba.
      - `leg > 0` flexiona el muslo (cadera) hacia adelante/arriba.
      - `knee < 0` flexiona la pantorrilla (rodilla) hacia atrás (+X respecto al muslo).
      - `ankle > 0` dorsiflexiona el pie (mantiene la suela plana en el suelo al doblar la rodilla).
    """
    tau = t % 22.0
    state = {j: 0.0 for j in JOINTS}

    if tau < 5.5:
        # Fase 1: Caminar hacia la izquierda (y: 0 -> +0.44 m, x: 0 -> -0.10 m)
        u = tau / 5.5
        s = smoothstep(u)
        vy = 0.44 * 6.0 * u * (1.0 - u)
        stride = 2.0 * math.pi * 1.6 * tau
        amp = math.sin(math.pi * u)

        swing_l = max(0.0, math.sin(stride - 0.35))
        swing_r = max(0.0, -math.sin(stride - 0.35))

        state['px'] = -0.10 * s
        state['py'] = 0.44 * s
        state['pz'] = 0.022 * abs(math.sin(stride)) * amp
        state['yaw'] = 0.35 * vy
        state['head_yaw'] = -0.75 * state['yaw']
        state['pitch'] = -0.05 * amp

        # Cinemática de piernas: cadera + flexión de rodilla en fase de balanceo + tobillo
        state['leg_l'] = 0.52 * math.sin(stride) * amp
        state['knee_l'] = (-0.82 * swing_l - 0.10) * amp
        state['ankle_l'] = (0.32 * swing_l - 0.20 * max(0.0, -math.sin(stride))) * amp

        state['leg_r'] = -0.52 * math.sin(stride) * amp
        state['knee_r'] = (-0.82 * swing_r - 0.10) * amp
        state['ankle_r'] = (0.32 * swing_r - 0.20 * max(0.0, math.sin(stride))) * amp

        # Cinemática de brazos: hombro alternado + flexión natural de codo
        state['arm_l'] = -0.48 * math.sin(stride) * amp
        state['elbow_l'] = 0.22 + 0.62 * max(0.0, -math.sin(stride)) * amp
        state['arm_r'] = 0.48 * math.sin(stride) * amp
        state['elbow_r'] = 0.22 + 0.62 * max(0.0, math.sin(stride)) * amp
        return state, 'walk_left'

    if tau < 10.0:
        # Fase 2: Sentadilla profunda con cinemática de cadena cerrada (pies apoyados en el suelo)
        u = (tau - 5.5) / 4.5
        if u < 0.25:
            crouch = smoothstep(u / 0.25)
        elif u < 0.75:
            crouch = 1.0 - 0.08 * math.sin(2.0 * math.pi * (u - 0.25) / 0.50)
        else:
            crouch = 1.0 - smoothstep((u - 0.75) / 0.25)

        sway = math.sin(2.0 * math.pi * (u - 0.2)) if 0.2 < u < 0.8 else 0.0
        state['px'] = -0.10 + 0.03 * crouch
        state['py'] = 0.44 - 0.06 * smoothstep(u) + 0.03 * sway
        state['pz'] = -0.26 * crouch
        state['yaw'] = -0.18 * sway
        state['head_yaw'] = 0.22 * sway
        state['pitch'] = -0.26 * crouch

        # Cadera (+1.02) + Rodilla (-1.62) + Tobillo (+0.60) = 0 rad (suelas planas en el piso)
        state['leg_l'] = 1.02 * crouch
        state['knee_l'] = -1.62 * crouch
        state['ankle_l'] = 0.60 * crouch
        state['leg_r'] = 1.02 * crouch
        state['knee_r'] = -1.62 * crouch
        state['ankle_r'] = 0.60 * crouch

        # Brazos al frente con codos flexionados en postura de equilibrio
        state['arm_l'] = 0.82 * crouch
        state['elbow_l'] = 0.92 * crouch
        state['arm_r'] = 0.82 * crouch
        state['elbow_r'] = 0.92 * crouch
        return state, 'crouch'

    if tau < 16.0:
        # Fase 3: Caminata fluida desde +0.38 m hasta -0.42 m (derecha)
        u = (tau - 10.0) / 6.0
        s = smoothstep(u)
        stride = 2.0 * math.pi * 1.8 * (tau - 10.0)
        amp = math.sin(math.pi * u)

        swing_l = max(0.0, math.sin(stride - 0.35))
        swing_r = max(0.0, -math.sin(stride - 0.35))

        state['px'] = -0.10 + 0.14 * math.sin(math.pi * u)
        state['py'] = 0.38 - 0.80 * s
        state['pz'] = 0.024 * abs(math.sin(stride)) * amp
        state['yaw'] = -0.42 * amp
        state['head_yaw'] = -0.75 * state['yaw']
        state['pitch'] = -0.06 * amp

        state['leg_l'] = 0.55 * math.sin(stride) * amp
        state['knee_l'] = (-0.85 * swing_l - 0.10) * amp
        state['ankle_l'] = (0.34 * swing_l - 0.22 * max(0.0, -math.sin(stride))) * amp

        state['leg_r'] = -0.55 * math.sin(stride) * amp
        state['knee_r'] = (-0.85 * swing_r - 0.10) * amp
        state['ankle_r'] = (0.34 * swing_r - 0.22 * max(0.0, math.sin(stride))) * amp

        state['arm_l'] = -0.50 * math.sin(stride) * amp
        state['elbow_l'] = 0.22 + 0.65 * max(0.0, -math.sin(stride)) * amp
        state['arm_r'] = 0.50 * math.sin(stride) * amp
        state['elbow_r'] = 0.22 + 0.65 * max(0.0, math.sin(stride)) * amp
        return state, 'walk_right'

    # Fase 4: Regreso hacia el centro + salto atlético con impulso de rodillas/tobillos y codos
    u = (tau - 16.0) / 6.0
    s = smoothstep(min(1.0, u / 0.35))
    state['px'] = -0.05 * s
    state['py'] = -0.42 * (1.0 - 0.65 * s)

    ju = (tau - 16.0) / 6.0
    if ju < 0.18:
        # Impulso (contra-movimiento: flexión de caderas, rodillas y tobillos, brazos atrás)
        k = math.sin(math.pi * (ju / 0.18))
        state['pz'] = -0.16 * k
        state['pitch'] = -0.22 * k
        state['arm_l'] = -0.55 * k
        state['elbow_l'] = 0.65 * k
        state['arm_r'] = -0.55 * k
        state['elbow_r'] = 0.65 * k
        state['leg_l'] = 0.72 * k
        state['knee_l'] = -1.25 * k
        state['ankle_l'] = 0.53 * k
        state['leg_r'] = 0.72 * k
        state['knee_r'] = -1.25 * k
        state['ankle_r'] = 0.53 * k
    elif ju < 0.72:
        # Vuelo en el aire + brazos arriba + flexión atlética de rodillas y extensión plantar
        air = (ju - 0.18) / 0.54
        lift = math.sin(math.pi * air) ** 0.65
        state['pz'] = 0.27 * lift
        state['pitch'] = 0.04 * lift
        state['arm_l'] = 2.05 * lift
        state['elbow_l'] = 0.45 * lift
        state['arm_r'] = 2.05 * lift
        state['elbow_r'] = 0.45 * lift
        state['leg_l'] = 0.40 * lift
        state['knee_l'] = -0.75 * lift
        state['ankle_l'] = -0.35 * lift
        state['leg_r'] = 0.25 * lift
        state['knee_r'] = -0.55 * lift
        state['ankle_r'] = -0.35 * lift
    elif ju < 0.90:
        # Aterrizaje amortiguado con flexión de rodillas, caderas y codos
        land = math.sin(math.pi * ((ju - 0.72) / 0.18))
        state['pz'] = -0.14 * land
        state['pitch'] = -0.18 * land
        state['arm_l'] = 0.50 * land
        state['elbow_l'] = 0.75 * land
        state['arm_r'] = 0.50 * land
        state['elbow_r'] = 0.75 * land
        state['leg_l'] = 0.62 * land
        state['knee_l'] = -1.10 * land
        state['ankle_l'] = 0.48 * land
        state['leg_r'] = 0.62 * land
        state['knee_r'] = -1.10 * land
        state['ankle_r'] = 0.48 * land
    return state, 'jump'


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    rclpy.init()
    node = Node('walk_person')

    pubs = {
        j: node.create_publisher(Float64, f'/person_target/{j}', 10)
        for j in JOINTS
    }
    latest = {}
    node.create_subscription(
        Image, '/person_tracker/debug_image', lambda m: latest.update(dbg=m), 2)
    node.create_subscription(
        PointStamped, '/person_tracker/target_point', lambda m: latest.update(pt=m), 2)

    def publish_state(st: dict[str, float]):
        for j, val in st.items():
            msg = Float64()
            msg.data = float(val)
            pubs[j].publish(msg)

    def save_snap(name: str):
        if 'dbg' in latest:
            m = latest['dbg']
            arr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
            cv2.imwrite(f'/home/ubuntu/.ros/lite6_handeye/{name}.png', arr)

    neutral = {j: 0.0 for j in JOINTS}
    t0 = time.time()
    while time.time() - t0 < 12.0 and 'pt' not in latest:
        publish_state(neutral)
        rclpy.spin_once(node, timeout_sec=0.03)
    t_hold = time.time()
    while time.time() - t_hold < 1.2:
        publish_state(neutral)
        rclpy.spin_once(node, timeout_sec=0.02)
    save_snap('sim_track_center')

    snaps_done = set()
    t_start = time.time()
    dt = 0.02  # 50 Hz
    while time.time() - t_start < duration:
        t = time.time() - t_start
        st, phase = choreography(t)
        publish_state(st)

        t_next = time.time() + dt
        while time.time() < t_next:
            rclpy.spin_once(node, timeout_sec=0.005)

        tau = t % 22.0
        if phase == 'walk_left' and 2.5 < tau < 3.0 and 'walk' not in snaps_done:
            save_snap('sim_track_walk')
            snaps_done.add('walk')
        elif phase == 'walk_left' and 4.8 < tau < 5.3 and 'left' not in snaps_done:
            save_snap('sim_track_left')
            snaps_done.add('left')
        elif phase == 'crouch' and 7.5 < tau < 8.2 and 'crouch' not in snaps_done:
            save_snap('sim_track_crouch')
            snaps_done.add('crouch')
        elif phase == 'walk_right' and 15.2 < tau < 15.8 and 'right' not in snaps_done:
            save_snap('sim_track_right')
            snaps_done.add('right')
        elif phase == 'jump' and 18.6 < tau < 19.2 and 'jump' not in snaps_done:
            save_snap('sim_track_jump')
            snaps_done.add('jump')

    publish_state(neutral)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
