from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w=5):
    if w <= 1:
        return np.asarray(x)
    x = np.asarray(x, dtype=float)
    if x.size < w:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[w:] - c[:-w]) / float(w)
    # pad to original length (left pad with first value)
    pad = np.full(w-1, y[0])
    return np.concatenate([pad, y])

def style_axes(ax, xlabel=None, ylabel=None, grid=True, ylim=None, legend_loc="best"):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if grid: ax.grid(True, alpha=0.35)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.legend(loc=legend_loc)
    plt.tight_layout()
    return ax
