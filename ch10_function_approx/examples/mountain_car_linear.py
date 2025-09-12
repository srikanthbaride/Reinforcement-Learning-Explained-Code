from __future__ import annotations
import numpy as np
from ..envs.mountain_car import MountainCar, MountainCarConfig
from ..features.tile_coding import TileCoder, ActionBlockTileCoder
from ..agents.linear_sarsa import LinearSarsaAgent

def make_tilecoder(n_tilings=8, bins=(8,8)):
    lows = np.array([-1.2, -0.07], dtype=float)
    highs = np.array([0.6, 0.07], dtype=float)
    offsets = []
    rng = np.random.default_rng(0)
    for t in range(n_tilings):
        offsets.append(rng.random(2) * 0.999)
    tc = TileCoder(lows=lows, highs=highs, bins_per_dim=bins, n_tilings=n_tilings, offsets=offsets)
    return tc

def run(episodes=50, seed=0, n_tilings=8):
    env = MountainCar(MountainCarConfig())
    tc = make_tilecoder(n_tilings=n_tilings, bins=(8,8))
    atc = ActionBlockTileCoder(tc, n_actions=env.nA)
    agent = LinearSarsaAgent(d=atc.d, nA=env.nA, gamma=1.0, alpha=0.5/n_tilings, eps=0.05, seed=seed)

    steps_per_ep = []
    for ep in range(episodes):
        s = env.reset(seed + ep)
        a = agent.act(atc.phi, s)
        steps = 0
        while True:
            ns, r, done, _ = env.step(a)
            na = agent.act(atc.phi, ns)
            agent.step(atc.phi, s, a, r, ns, na)
            s, a = ns, na
            steps += 1
            if done: break
        steps_per_ep.append(steps)
    return np.array(steps_per_ep), agent.w

if __name__ == "__main__":
    steps, w = run(episodes=20, seed=123, n_tilings=8)
    print("Steps per episode:", steps)
