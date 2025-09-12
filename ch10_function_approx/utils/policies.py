from __future__ import annotations
import numpy as np

def epsilon_greedy(Q_row: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(Q_row.size))
    return int(np.argmax(Q_row))
