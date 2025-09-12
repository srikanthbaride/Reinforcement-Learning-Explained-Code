from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MountainCarConfig:
    x_min: float = -1.2
    x_max: float = 0.6
    v_min: float = -0.07
    v_max: float = 0.07
    goal_x: float = 0.5
    gamma: float = 1.0
    max_steps: int = 2000

class MountainCar:
    LEFT, NEUTRAL, RIGHT = 0, 1, 2

    def __init__(self, cfg: MountainCarConfig = MountainCarConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng()
        self.reset()

    @property
    def nA(self): return 3

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = self.rng.uniform(-0.6, -0.4)
        self.v = 0.0
        self.t = 0
        return np.array([self.x, self.v], dtype=float)

    def step(self, a: int):
        assert 0 <= a < self.nA
        force = {self.LEFT: -1.0, self.NEUTRAL: 0.0, self.RIGHT: +1.0}[a]
        v = self.v + 0.001 * force - 0.0025 * np.cos(3 * self.x)
        v = np.clip(v, self.cfg.v_min, self.cfg.v_max)
        x = self.x + v
        if x < self.cfg.x_min:
            x = self.cfg.x_min
            v = 0.0
        self.x, self.v = x, v
        self.t += 1
        done = (self.x >= self.cfg.goal_x) or (self.t >= self.cfg.max_steps)
        reward = 0.0 if (self.x >= self.cfg.goal_x) else -1.0
        return np.array([self.x, self.v], dtype=float), reward, done, {}

    def state(self):
        return np.array([self.x, self.v], dtype=float)
