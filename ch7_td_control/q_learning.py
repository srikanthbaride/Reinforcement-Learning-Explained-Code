# ch7_td_control/q_learning.py
from __future__ import annotations
import random
from typing import Dict, Tuple, Callable, List
from .policies import eps_greedy_action

State = Tuple[int, int]
QTable = Dict[Tuple[State, int], float]

def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, eps_schedule=lambda t:0.1, seed=0):
    if seed is not None: random.seed(seed)
    Q: QTable = {}
    returns: List[float] = []
    for ep in range(episodes):
        s = env.reset()
        done, G = False, 0.0
        while not done:
            a = eps_greedy_action(Q, s, env.action_space.n, eps_schedule(ep))
            s2, r, done, _ = env.step(a)
            G += r
            if not done:
                best_next = max(Q.get((s2, a2), 0.0) for a2 in range(env.action_space.n))
                target = r + gamma * best_next
            else:
                target = r
            Q[(s, a)] = Q.get((s, a), 0.0) + alpha * (target - Q.get((s, a), 0.0))
            s = s2
        returns.append(G)
    return Q, returns
