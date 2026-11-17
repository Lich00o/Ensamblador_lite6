"""Panel de control del person_tracker (ventana Tk, dos pestañas).

Pestaña "Seguimiento":

  [ACTIVAR / PAUSAR seguimiento]   -> /person_tracker/set_tracking (std_srvs/SetBool)
  [Ir a pose de vigilancia]        -> /person_tracker/go_watch     (std_srvs/Trigger)
  [Robot a mode 1 (listo)]         -> /ufactory/set_mode 1 + set_state 0 y reactiva
                                      lite6_traj_controller (el driver a veces no lo reactiva
                                      si el robot vuelve a 'ready' durante la desactivación)
  Objetivo: un botón por ID visible + [Más cercana] [Soltar] -> /person_tracker/select_target
  (también se elige con clic sobre la persona en rqt_image_view de /person_tracker/debug_image)
Pestaña "Mover robot" (jog.py): botones J1-J6 y X/Y/Z/Rx/Ry/Rz que mueven mientras se
mantienen presionados, velocidad en %, ángulos y pose del flange. Al empezar a mover se pausa
el seguimiento.
Abajo: estado del robot (mode/state/err).
Atajo: barra espaciadora = activar/pausar.

Uso (dentro del contenedor, con el tracker corriendo): python3 tracker_panel.py
"""
import json
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import SetBool, Trigger
from xarm_msgs.msg import RobotMsg
from controller_manager_msgs.srv import SwitchController
from xarm_msgs.srv import SetInt16

from jog import Jogger

NS = '/person_tracker'


class PanelNode(Node):
    def __init__(self):
        super().__init__('tracker_panel')
        self.tracking = None
        self.robot = None
        self.msg = ''
        self.tracks, self.target = [], None
        self.create_subscription(String, f'{NS}/tracks', self._on_tracks, 5)
        self.pub_sel = self.create_publisher(Int32, f'{NS}/select_target', 10)
        self.create_subscription(Bool, f'{NS}/tracking_enabled', self._on_tracking,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(RobotMsg, '/ufactory/robot_states', self._on_robot, 10)
        self.cli_track = self.create_client(SetBool, f'{NS}/set_tracking')
        self.cli_watch = self.create_client(Trigger, f'{NS}/go_watch')
        self.cli_mode = self.create_client(SetInt16, '/ufactory/set_mode')
        self.cli_state = self.create_client(SetInt16, '/ufactory/set_state')
        self.cli_switch = self.create_client(SwitchController, '/controller_manager/switch_controller')

    def _on_tracking(self, m):
        self.tracking = m.data

    def _on_tracks(self, m):
        d = json.loads(m.data)
        self.tracks, self.target = d['tracks'], d['target']

    def select(self, tid):
        self.pub_sel.publish(Int32(data=int(tid)))

    def _on_robot(self, m):
        self.robot = (m.mode, m.state, m.err)

    def _call(self, cli, req, label):
        if not cli.service_is_ready():
            self.msg = f'{label}: servicio no disponible (¿tracker corriendo?)'
            return
        fut = cli.call_async(req)
        fut.add_done_callback(lambda f: setattr(self, 'msg', f'{label}: {getattr(f.result(), "message", "ok")}'))

    def toggle(self):
        self._call(self.cli_track, SetBool.Request(data=not bool(self.tracking)), 'seguimiento')

    def go_watch(self):
        self._call(self.cli_watch, Trigger.Request(), 'vigilancia')

    def pause_tracking(self):
        if self.tracking:
            self._call(self.cli_track, SetBool.Request(data=False), 'seguimiento')

    def robot_ok(self):
        return self.robot is not None and self.robot[0] == 1 and self.robot[2] == 0 \
            and self.robot[1] in (0, 1, 2)

    def robot_ready(self):
        self._call(self.cli_mode, SetInt16.Request(data=1), 'mode 1')
        self._call(self.cli_state, SetInt16.Request(data=0), 'state 0')

        def activate():
            time.sleep(1.5)                  # el driver desactiva/reactiva al cambiar de modo
            req = SwitchController.Request(activate_controllers=['lite6_traj_controller'],
                                           strictness=1, activate_asap=True)
            self._call(self.cli_switch, req, 'controlador')
        threading.Thread(target=activate, daemon=True).start()


def build_jog_tab(parent, jog):
    """Botones de jog (mantener presionado) + velocidad + pose actual."""
    f_spd = tk.Frame(parent)
    f_spd.pack(fill='x', padx=10, pady=(8, 2))
    tk.Label(f_spd, text='Velocidad %').pack(side='left')
    spd = tk.Scale(f_spd, from_=5, to=100, orient='horizontal', resolution=5,
                   command=lambda v: setattr(jog, 'speed', float(v) / 100.0))
    spd.set(int(jog.speed * 100))
    spd.pack(side='left', fill='x', expand=True)

    def hold(btn, axis, sign):
        btn.bind('<ButtonPress-1>', lambda e: jog.press(axis, sign))
        btn.bind('<ButtonRelease-1>', lambda e: jog.release())
        btn.bind('<Leave>', lambda e: jog.release())

    grid = tk.Frame(parent)
    grid.pack(padx=10, pady=4)
    tk.Label(grid, text='Articular', font=('DejaVu Sans', 10, 'bold')).grid(row=0, column=0, columnspan=3)
    tk.Label(grid, text='Cartesiano (base)', font=('DejaVu Sans', 10, 'bold')).grid(row=0, column=4, columnspan=3)
    cart = [('X', 'x'), ('Y', 'y'), ('Z', 'z'), ('Rx', 'rx'), ('Ry', 'ry'), ('Rz', 'rz')]
    for i in range(6):
        for col, sign, txt in ((0, -1, '−'), (2, 1, '+')):
            b = tk.Button(grid, text=f'J{i + 1} {txt}', width=6, height=1)
            b.grid(row=i + 1, column=col, padx=2, pady=2)
            hold(b, ('j', i), sign)
        name, key = cart[i]
        for col, sign, txt in ((4, -1, '−'), (6, 1, '+')):
            b = tk.Button(grid, text=f'{name} {txt}', width=6, height=1)
            b.grid(row=i + 1, column=col, padx=2, pady=2)
            hold(b, (key, None), sign)
    tk.Label(grid, width=2).grid(row=1, column=3)
    tk.Label(parent, text='Mantén presionado para mover; al soltar frena suave.\n'
             'Al empezar a mover se pausa el seguimiento.', fg='#555',
             font=('DejaVu Sans', 9)).pack(pady=2)
    l_pose = tk.Label(parent, font=('DejaVu Sans Mono', 10), justify='left')
    l_pose.pack(pady=6)
    return l_pose


def main():
    rclpy.init()
    node = PanelNode()
    jog = Jogger(node, on_start=node.pause_tracking)
    jog.robot_ok = node.robot_ok
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    root = tk.Tk()
    root.title('Tracker de personas')
    root.geometry('430x560')
    big = ('DejaVu Sans', 16, 'bold')
    nb = ttk.Notebook(root)
    nb.pack(fill='both', expand=True)
    tab = tk.Frame(nb)
    tab_jog = tk.Frame(nb)
    nb.add(tab, text='Seguimiento')
    nb.add(tab_jog, text='Mover robot')
    b_track = tk.Button(tab, font=big, height=2, command=node.toggle)
    b_track.pack(fill='x', padx=12, pady=(12, 6))
    tk.Button(tab, text='Ir a pose de vigilancia', font=('DejaVu Sans', 12),
              command=node.go_watch).pack(fill='x', padx=12, pady=4)
    tk.Button(tab, text='Robot a mode 1 (listo)', font=('DejaVu Sans', 12),
              command=node.robot_ready).pack(fill='x', padx=12, pady=4)
    tk.Label(tab, text='Objetivo (o clic sobre la persona en la imagen):',
             font=('DejaVu Sans', 10)).pack(pady=(10, 0))
    f_ids = tk.Frame(tab)
    f_ids.pack(fill='x', padx=12, pady=4)
    f_sel = tk.Frame(tab)
    f_sel.pack(fill='x', padx=12)
    tk.Button(f_sel, text='Más cercana', command=lambda: node.select(-1)).pack(side='left', expand=True, fill='x')
    tk.Button(f_sel, text='Soltar objetivo', command=lambda: node.select(-2)).pack(side='left', expand=True, fill='x')
    l_target = tk.Label(tab, font=('DejaVu Sans', 12, 'bold'))
    l_target.pack(pady=4)
    shown = [None]                       # fuerza el primer dibujado
    l_robot = tk.Label(root, font=('DejaVu Sans', 12))
    l_robot.pack(pady=(10, 2))
    l_msg = tk.Label(root, font=('DejaVu Sans', 10), fg='#555', wraplength=330)
    l_msg.pack(pady=2)
    root.bind('<space>', lambda e: node.toggle())
    l_pose = build_jog_tab(tab_jog, jog)

    def refresh():
        if node.tracking is None:
            b_track.config(text='esperando tracker...', bg='#bbbbbb', activebackground='#bbbbbb')
        elif node.tracking:
            b_track.config(text='SEGUIMIENTO ON  (pausar)', bg='#3cb371', activebackground='#2e8b57')
        else:
            b_track.config(text='PAUSA  (activar)', bg='#e0a030', activebackground='#c08020')
        if node.robot is None:
            l_robot.config(text='robot: sin estado', fg='#999')
        else:
            mode, state, err = node.robot
            ok = mode == 1 and err == 0 and state in (0, 1, 2)
            l_robot.config(text=f'robot: mode {mode}  state {state}  err {err}' + ('' if ok else '  (NO LISTO)'),
                           fg='#2e8b57' if ok else '#cc0000')
        vis = [t['id'] for t in node.tracks if t['visible']]
        if vis != shown:                     # rehacer botones solo si cambian los IDs visibles
            for w in f_ids.winfo_children():
                w.destroy()
            for tid in vis:
                tk.Button(f_ids, text=f'ID {tid}', width=6,
                          command=lambda t=tid: node.select(t)).pack(side='left', padx=2)
            if not vis:
                tk.Label(f_ids, text='(nadie a la vista)', fg='#999').pack(side='left')
            shown[:] = vis
        for w in f_ids.winfo_children():
            if isinstance(w, tk.Button):
                is_t = w.cget('text') == f'ID {node.target}'
                w.config(bg='#e04040' if is_t else '#dddddd', fg='white' if is_t else 'black')
        if node.target is None:
            l_target.config(text='Sin objetivo', fg='#c08020')
        else:
            lost = node.target not in vis
            l_target.config(text=f'Objetivo: ID {node.target}' + ('  (PERDIDO)' if lost else ''),
                            fg='#cc0000' if lost else '#2e8b57')
        l_msg.config(text=jog.msg or node.msg)
        jog.keepalive()
        tcp = jog.tcp()
        if tcp is not None:
            p, rpy = tcp
            qd = np.degrees(jog.q_meas)
            l_pose.config(text='J ' + '  '.join(f'{v:6.1f}' for v in qd) +
                          f'\nflange  x {p[0]:.0f}  y {p[1]:.0f}  z {p[2]:.0f} mm'
                          f'\nrx {rpy[0]:.1f}  ry {rpy[1]:.1f}  rz {rpy[2]:.1f} °')
        root.after(100, refresh)

    refresh()
    root.mainloop()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
