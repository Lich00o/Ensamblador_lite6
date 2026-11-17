"""Suavizado de la consigna articular para seguimiento en línea.

El tracker produce metas articulares a ~12 Hz con saltos; enviarlas como trayectorias de un
punto (velocidad final 0) hace que el brazo frene y arranque en cada una. `JointSmoother`
genera en línea, a la frecuencia de envío (p. ej. 50 Hz), una referencia continua en posición,
velocidad y aceleración que persigue la última meta respetando límites de velocidad,
aceleración y jerk (Ruckig, generación de trayectorias online). Sin `ruckig` instalado usa un
seguidor de segundo orden con velocidad y aceleración limitadas (sin límite de jerk).
"""

from typing import Tuple

import numpy as np

try:
    import ruckig
except ImportError:  # pragma: no cover - depende del entorno
    ruckig = None


class JointSmoother:
    def __init__(self, n: int, dt: float, max_vel: float, max_acc: float, max_jerk: float,
                 use_ruckig: bool = True):
        self.n, self.dt = int(n), float(dt)
        self.vmax, self.amax, self.jmax = float(max_vel), float(max_acc), float(max_jerk)
        self.backend = 'ruckig' if (use_ruckig and ruckig is not None) else 'acc-limited'
        if self.backend == 'ruckig':
            self._otg = ruckig.Ruckig(self.n, self.dt)
            self._inp = ruckig.InputParameter(self.n)
            self._out = ruckig.OutputParameter(self.n)
            self._inp.max_velocity = [self.vmax] * self.n
            self._inp.max_acceleration = [self.amax] * self.n
            self._inp.max_jerk = [self.jmax] * self.n
        self.q = self.v = self.a = self.goal = None

    def set_limits(self, max_vel=None, max_acc=None, max_jerk=None):
        """Cambia los límites en caliente (se aplican desde el próximo paso)."""
        self.vmax = float(max_vel) if max_vel is not None else self.vmax
        self.amax = float(max_acc) if max_acc is not None else self.amax
        self.jmax = float(max_jerk) if max_jerk is not None else self.jmax
        if self.backend == 'ruckig':
            self._inp.max_velocity = [self.vmax] * self.n
            self._inp.max_acceleration = [self.amax] * self.n
            self._inp.max_jerk = [self.jmax] * self.n

    @property
    def active(self) -> bool:
        return self.q is not None

    def reset(self, q, v=None):
        """Arranca desde el estado medido (velocidad 0 si no se conoce)."""
        self.q = np.asarray(q, dtype=float).copy()
        self.v = np.zeros(self.n) if v is None else np.asarray(v, dtype=float).copy()
        self.a = np.zeros(self.n)
        self.goal = self.q.copy()

    def stop(self):
        self.q = self.v = self.a = self.goal = None

    def set_goal(self, q_goal):
        self.goal = np.asarray(q_goal, dtype=float).copy()

    def settled(self, tol: float = 1e-4) -> bool:
        return (self.active and float(np.max(np.abs(self.goal - self.q))) < tol
                and float(np.max(np.abs(self.v))) < tol)

    def step(self, lookahead: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Avanza `dt`. Devuelve (q, v) actuales y (q, v) muestreados `lookahead` s más adelante."""
        if self.backend == 'ruckig':
            inp, out = self._inp, self._out
            inp.current_position, inp.current_velocity = self.q.tolist(), self.v.tolist()
            inp.current_acceleration = self.a.tolist()
            inp.target_position = self.goal.tolist()
            inp.target_velocity, inp.target_acceleration = [0.0] * self.n, [0.0] * self.n
            res = self._otg.update(inp, out)
            if res not in (ruckig.Result.Working, ruckig.Result.Finished):
                raise RuntimeError(f'ruckig: {res}')
            self.q = np.array(out.new_position)
            self.v = np.array(out.new_velocity)
            self.a = np.array(out.new_acceleration)
            t_la = min(out.time + lookahead, out.trajectory.duration)
            qa, va, _ = out.trajectory.at_time(t_la)
            if out.time + lookahead > out.trajectory.duration:
                qa, va = self.goal.tolist(), [0.0] * self.n
            return self.q.copy(), self.v.copy(), np.array(qa), np.array(va)
        # seguidor de 2.º orden: velocidad deseada con frenado a aceleración máxima
        e = self.goal - self.q
        adt = self.amax * self.dt                     # perfil de frenado discreto (llega con v -> 0)
        v_des = np.sign(e) * np.minimum(
            self.vmax, adt * (np.sqrt(0.25 + 2.0 * np.abs(e) / (adt * self.dt)) - 0.5))
        dv = np.clip(v_des - self.v, -self.amax * self.dt, self.amax * self.dt)
        self.a = dv / self.dt
        self.v = self.v + dv
        step = self.v * self.dt
        over = np.abs(step) > np.abs(e)               # no sobrepasar la meta
        step[over] = e[over]
        self.v[over] = 0.0
        self.q = self.q + step
        qa = self.q + self.v * lookahead
        return self.q.copy(), self.v.copy(), qa, self.v.copy()

