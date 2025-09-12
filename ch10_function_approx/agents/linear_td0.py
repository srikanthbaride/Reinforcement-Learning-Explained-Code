from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class LinearTD0:
    d: int
    gamma: float = 0.99
    alpha: float = 0.1
    seed: int | None = None

    def __post_init__(self):
        self.w = np.zeros(self.d, dtype=float)
        self.rng = np.random.default_rng(self.seed)

    def predict(self, phi_s: np.ndarray) -> float:
        return float(self.w @ phi_s)

    def update(self, phi_s: np.ndarray, r: float, phi_ns: np.ndarray):
        delta = r + self.gamma * (self.w @ phi_ns) - (self.w @ phi_s)
        self.w += self.alpha * delta * phi_s
        return delta
