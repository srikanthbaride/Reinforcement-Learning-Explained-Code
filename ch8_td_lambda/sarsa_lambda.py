from __future__ import annotations
from typing import Optional
import numpy as np

def sarsa_lambda_control(
    env,
    gamma: float = 0.99,
    alpha: float = 0.1,
    lam: float = 0.9,
    epsilon: float = 0.1,
    episodes: int = 1000,
    n_states: Optional[int] = None,
    n_actions: Optional[int] = None,
    trace_type: str = 'accumulating',
    seed: Optional[int] = None,
) -> np.ndarray:
    '''
    On-policy SARSA(lambda) with eligibility traces (tabular Q).
    '''
    rng = np.random.default_rng(seed)
    if n_states is None:
        n_states = getattr(env, 'n_states', getattr(env, 'nS', None))
        if n_states is None:
            raise ValueError('Provide n_states or ensure env has n_states/nS.')
    if n_actions is None:
        n_actions = getattr(env, 'n_actions', getattr(env, 'nA', None))
        if n_actions is None:
            raise ValueError('Provide n_actions or ensure env has n_actions/nA.')

    Q = np.zeros((n_states, n_actions), dtype=float)

    def eps_greedy(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_actions))
        return int(np.argmax(Q[s]))

    for _ in range(episodes):
        E = np.zeros_like(Q)  # eligibility for (s,a)
        s = env.reset()
        a = eps_greedy(s)

        while True:
            s_next, r, done, *_ = env.step(a)
            a_next = 0 if done else eps_greedy(s_next)

            q_next = 0.0 if done else Q[s_next, a_next]
            delta = r + gamma * q_next - Q[s, a]

            # decay all, then reinforce current pair
            E *= (gamma * lam)
            if trace_type == 'replacing':
                E[s, :] = 0.0
                E[s, a] = 1.0
            elif trace_type == 'accumulating':
                E[s, a] += 1.0
            else:
                raise ValueError("trace_type must be 'accumulating' or 'replacing'.")

            Q += alpha * delta * E

            s, a = s_next, a_next
            if done:
                break

    return Q
