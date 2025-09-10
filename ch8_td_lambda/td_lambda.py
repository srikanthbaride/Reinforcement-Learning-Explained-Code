from __future__ import annotations
from typing import Callable, Union, Sequence, Optional
import numpy as np

Action = int
Probs = Union[Sequence[float], np.ndarray]
Policy = Callable[[int], Union[Action, Probs]]

def _sample_action(act_or_probs: Union[Action, Probs], rng: np.random.Generator) -> Action:
    if isinstance(act_or_probs, (list, tuple, np.ndarray)):
        p = np.asarray(act_or_probs, dtype=float)
        if p.ndim != 1:
            raise ValueError('Policy probabilities must be 1-D.')
        s = p.sum()
        if s <= 0:
            raise ValueError('Non-positive probability vector from policy.')
        if not np.isclose(s, 1.0):
            p = p / s
        return int(rng.choice(len(p), p=p))
    return int(act_or_probs)

def td_lambda_prediction(
    env,
    policy: Policy,
    gamma: float = 0.99,
    alpha: float = 0.1,
    lam: float = 0.9,
    episodes: int = 500,
    n_states: Optional[int] = None,
    trace_type: str = 'accumulating',
    seed: Optional[int] = None,
) -> np.ndarray:
    '''
    On-policy TD(lambda) prediction with eligibility traces (tabular, backward view).
    '''
    rng = np.random.default_rng(seed)
    if n_states is None:
        n_states = getattr(env, 'n_states', getattr(env, 'nS', None))
        if n_states is None:
            raise ValueError('Provide n_states or ensure env has n_states/nS.')

    V = np.zeros(n_states, dtype=float)

    for _ in range(episodes):
        e = np.zeros_like(V)
        s = env.reset()

        while True:
            a = _sample_action(policy(s), rng)
            s_next, r, done, *_ = env.step(a)
            v_next = 0.0 if done else V[s_next]
            delta = r + gamma * v_next - V[s]

            # traces: decay then reinforce
            e *= (gamma * lam)
            if trace_type == 'replacing':
                e[s] = 1.0
            elif trace_type == 'accumulating':
                e[s] += 1.0
            else:
                raise ValueError("trace_type must be 'accumulating' or 'replacing'.")

            V += alpha * delta * e
            s = s_next
            if done:
                break

    return V
