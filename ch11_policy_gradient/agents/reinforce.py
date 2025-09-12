from dataclasses import dataclass
import numpy as np
from typing import NamedTuple, Callable, Optional
from ..utils.returns import returns_to_go, standardize

class Trajectory(NamedTuple):
    states: list
    actions: list
    rewards: list
    logps: list

@dataclass
class Reinforce:
    gamma: float = 1.0
    alpha: float = 0.05
    normalize_adv: bool = True
    baseline_fn: Optional[Callable[[object], float]] = None
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def run_episode_discrete(self, env, policy, feature_fn: Callable[[object], np.ndarray]):
        s = env.reset()
        S, A, R, L = [], [], [], []
        done = False
        while not done:
            x = feature_fn(s)
            a = policy.sample(x)
            logp, _ = policy.logprob_and_grad(x, a)
            ns, r, done, _ = env.step(a)
            S.append(s); A.append(a); R.append(r); L.append(logp)
            s = ns
        return Trajectory(S, A, R, L)

    def update_discrete(self, traj: Trajectory, policy, feature_fn: Callable[[object], np.ndarray]):
        G = returns_to_go(traj.rewards, self.gamma)
        if self.baseline_fn is not None:
            b = np.array([self.baseline_fn(s) for s in traj.states], dtype=float)
            adv = G - b
        else:
            adv = G.copy()
        if self.normalize_adv:
            if len(adv) >= 2 and np.std(adv) > 1e-8:
                adv = standardize(adv)

        total_grad = np.zeros_like(policy.theta)
        for s, a, adv_t in zip(traj.states, traj.actions, adv):
            x = feature_fn(s)
            _, grad = policy.logprob_and_grad(x, a)
            total_grad += adv_t * grad
        policy.theta += self.alpha * total_grad
        return {"G": G, "adv": adv}
