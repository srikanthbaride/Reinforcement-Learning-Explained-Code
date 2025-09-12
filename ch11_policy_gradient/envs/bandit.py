import numpy as np
from dataclasses import dataclass

@dataclass
class TwoArmedBandit:
    q_star: tuple[float, float] = (1.0, 1.5)
    seed: int | None = None
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
    @property
    def nA(self): return 2
    def reset(self):
        return np.array([1.0], dtype=float)  # x(s)=1
    def step(self, a: int):
        assert a in (0,1)
        r = float(self.rng.normal(self.q_star[a], 1.0))
        return None, r, True, {}
