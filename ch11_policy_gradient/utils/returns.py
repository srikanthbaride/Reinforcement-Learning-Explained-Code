import numpy as np

def returns_to_go(rewards, gamma: float) -> np.ndarray:
    G = np.zeros(len(rewards), dtype=float)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        G[t] = g
    return G

def standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean()
    std = x.std()
    return (x - mu) / (std + eps)
