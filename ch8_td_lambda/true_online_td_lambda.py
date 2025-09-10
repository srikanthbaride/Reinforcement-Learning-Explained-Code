from __future__ import annotations
from typing import Callable, Optional
import numpy as np

FeatureMap = Callable[[int], np.ndarray]

def true_online_td_lambda_linear(
    env,
    phi: FeatureMap,
    gamma: float = 0.99,
    alpha: float = 0.1,
    lam: float = 0.9,
    episodes: int = 500,
    d: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    '''
    True Online TD(lambda) (van Seijen & Sutton, 2014) for on-policy prediction with linear features.
    '''
    rng = np.random.default_rng(seed)
    if d is None:
        d = int(phi(env.reset()).shape[0])
    w = np.zeros(d, dtype=float)
    z = np.zeros_like(w)

    for _ in range(episodes):
        z[:] = 0.0
        s = env.reset()
        phi_prev = np.zeros_like(w)  # φ_{t-1}; zero at start

        while True:
            # random behavior policy over env.n_actions if present
            nA = getattr(env, 'n_actions', getattr(env, 'nA', None))
            a = int(rng.integers(nA)) if nA is not None else 0
            s_next, r, done, *_ = env.step(a)

            phi_t = phi(s)
            v_t = np.dot(w, phi_t)
            phi_tp1 = np.zeros_like(phi_t) if done else phi(s_next)
            v_tp1 = 0.0 if done else np.dot(w, phi_tp1)
            delta = r + gamma * v_tp1 - v_t

            # dutch trace
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, phi_t)) * phi_t

            w += alpha * (delta + v_t - np.dot(w, phi_prev)) * z
            w -= alpha * (v_t - np.dot(w, phi_prev)) * phi_t

            phi_prev = phi_t
            s = s_next
            if done:
                break

    return w
