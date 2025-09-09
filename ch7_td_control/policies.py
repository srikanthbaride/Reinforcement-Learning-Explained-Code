# ch7_td_control/policies.py
from __future__ import annotations
import math, random
from typing import Callable, Dict, Tuple

State = Tuple[int, int]
QTable = Dict[Tuple[State, int], float]

def eps_greedy_action(q: QTable, s: State, n_actions: int, eps: float) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)
    best_a, best_q = 0, float("-inf")
    for a in range(n_actions):
        val = q.get((s, a), 0.0)
        if val > best_q:
            best_q, best_a = val, a
    return best_a

def fixed_epsilon(eps: float) -> Callable[[int], float]:
    return lambda t: eps

def linear_decay(eps0: float, eps_min: float, T: int) -> Callable[[int], float]:
    slope = (eps_min - eps0) / max(1, T)
    return lambda t: max(eps_min, eps0 + slope * t)

def exp_decay(eps0: float, k: float, eps_min: float = 0.0) -> Callable[[int], float]:
    return lambda t: max(eps_min, eps0 * math.exp(-k * t))

def inverse_glie(c: float = 1.0) -> Callable[[int], float]:
    return lambda t: c / (t + 1)
