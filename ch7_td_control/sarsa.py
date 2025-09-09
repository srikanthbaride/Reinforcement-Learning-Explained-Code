# ch7_td_control/sarsa.py
from __future__ import annotations
import random
from typing import Dict, Tuple, Callable, List
from .policies import eps_greedy_action

State = Tuple[int, int]
QTable = Dict[Tuple[State, int], float]

def sarsa(env, episodes=500, alpha=0.1, gamma=0.99, eps_schedule=lambda t:0.1, seed=0):
    if seed is not None: random.seed(seed)
    Q: QTable = {}
    returns: List[float] = []
    for ep in range(episodes):
        s = env.reset()
        eps = eps_schedule(ep)
        a = eps_greedy_action(Q, s, env.action_space.n, eps)
        done, G = False, 0.0
        while not done:
            s2, r, done, _ = env.step(a)
            G += r
            if not done:
                a2 = eps_greedy_action(Q, s2, env.action_space.n, eps_schedule(ep))
                target = r + gamma * Q.get((s2, a2), 0.0)
            else:
                a2, target = None, r
            Q[(s, a)] = Q.get((s, a), 0.0) + alpha * (target - Q.get((s, a), 0.0))
            s, a = s2, (a2 if a2 is not None else 0)
        returns.append(G)
    return Q, returns
