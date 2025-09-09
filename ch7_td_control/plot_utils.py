# ch7_td_control/plot_utils.py
from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt
from typing import List

def moving_average(x: List[float], k: int = 20):
    if k <= 1: return np.asarray(x, dtype=float)
    pad = np.pad(np.asarray(x, dtype=float), (k-1, 0), mode='edge')
    w = np.ones(k) / k
    return np.convolve(pad, w, mode='valid')

def plot_learning_curves(episodes, sarsa_ret, ql_ret, ma_k=20, title="Cliff-Walking: Average Return per Episode"):
    s_ma = moving_average(sarsa_ret, ma_k)
    q_ma = moving_average(ql_ret, ma_k)
    plt.figure(figsize=(8,4.8))
    plt.plot(episodes, s_ma, label=f"SARSA (MA k={ma_k})", linewidth=2)
    plt.plot(episodes, q_ma, label=f"Q-learning (MA k={ma_k})", linewidth=2)
    plt.xlabel("Episode"); plt.ylabel("Average return")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(); plt.title(title); plt.tight_layout(); plt.show()
