import numpy as np
from dataclasses import dataclass

@dataclass
class SoftmaxPolicy:
    nA: int
    d: int
    theta: np.ndarray | None = None
    seed: int | None = None
    def __post_init__(self):
        if self.theta is None:
            self.theta = np.zeros((self.nA, self.d), dtype=float)
        self.rng = np.random.default_rng(self.seed)
    def prefs(self, x: np.ndarray) -> np.ndarray:
        return self.theta @ x
    def probs(self, x: np.ndarray) -> np.ndarray:
        h = self.prefs(x); h -= np.max(h)
        e = np.exp(h); return e / e.sum()
    def sample(self, x: np.ndarray) -> int:
        p = self.probs(x); return int(self.rng.choice(self.nA, p=p))
    def logprob_and_grad(self, x: np.ndarray, a: int):
        p = self.probs(x); logp = float(np.log(p[a] + 1e-12))
        grad = -np.outer(p, x); grad[a, :] += x
        return logp, grad
