from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..utils.policies import epsilon_greedy

@dataclass
class LinearSarsaAgent:
    d: int
    nA: int
    gamma: float = 1.0
    alpha: float = 0.5
    eps: float = 0.05
    seed: int | None = None

    def __post_init__(self):
        self.w = np.zeros(self.d, dtype=float)
        self.rng = np.random.default_rng(self.seed)

    def q_row(self, phi_fn, s_vec) -> np.ndarray:
        vals = np.zeros(self.nA, dtype=float)
        for a in range(self.nA):
            vals[a] = self.w @ phi_fn(s_vec, a)
        return vals

    def act(self, phi_fn, s_vec) -> int:
        q = self.q_row(phi_fn, s_vec)
        return epsilon_greedy(q, self.eps, self.rng)

    def step(self, phi_fn, s_vec, a, r, ns_vec, na):
        phi_sa = phi_fn(s_vec, a)
        phi_ns_na = phi_fn(ns_vec, na)
        td_target = r + self.gamma * (self.w @ phi_ns_na)
        td_err = td_target - (self.w @ phi_sa)
        self.w += self.alpha * td_err * phi_sa
        return td_err
