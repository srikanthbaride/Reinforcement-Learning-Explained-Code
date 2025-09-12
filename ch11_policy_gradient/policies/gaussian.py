import numpy as np
from dataclasses import dataclass

@dataclass
class GaussianPolicy1D:
    mu: float = 0.0
    log_sigma: float = 0.0
    seed: int | None = None
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
    @property
    def sigma(self) -> float:
        return float(np.exp(self.log_sigma))
    def sample(self, _x=None) -> float:
        return float(self.rng.normal(self.mu, self.sigma))
    def logprob_and_grad(self, a: float, _x=None):
        sigma2 = self.sigma ** 2
        logp = -0.5 * ((a - self.mu) ** 2 / sigma2 + np.log(2*np.pi*sigma2))
        dmu = (a - self.mu) / sigma2
        dlogs = ((a - self.mu) ** 2) / sigma2 - 1.0
        return float(logp), np.array([dmu, dlogs], dtype=float)
