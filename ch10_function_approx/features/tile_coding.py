from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple, List

@dataclass
class TileCoder:
    lows: np.ndarray
    highs: np.ndarray
    bins_per_dim: Tuple[int, ...]
    n_tilings: int
    offsets: List[np.ndarray]

    def __post_init__(self):
        self.lows = np.asarray(self.lows, dtype=float)
        self.highs = np.asarray(self.highs, dtype=float)
        assert self.lows.shape == self.highs.shape
        self.d = self.lows.size
        assert len(self.bins_per_dim) == self.d
        assert len(self.offsets) == self.n_tilings
        self.bins_per_dim = tuple(int(b) for b in self.bins_per_dim)
        self.tiles_per_tiling = int(np.prod(self.bins_per_dim))
        self.total_tiles = self.tiles_per_tiling * self.n_tilings
        self.bin_sizes = (self.highs - self.lows) / np.array(self.bins_per_dim, dtype=float)

    def _coord_to_index_single(self, x: np.ndarray, tiling_idx: int) -> int:
        offs = self.offsets[tiling_idx] * self.bin_sizes
        z = (x - (self.lows - offs)) / self.bin_sizes
        idxs = np.floor(z).astype(int)
        idxs = np.clip(idxs, 0, np.array(self.bins_per_dim) - 1)
        flat = 0
        for i, b in enumerate(self.bins_per_dim):
            flat = flat * b + idxs[i]
        return flat

    def active_indices(self, x: Iterable[float]) -> List[int]:
        x = np.asarray(x, dtype=float)
        inds = []
        for t in range(self.n_tilings):
            local = self._coord_to_index_single(x, t)
            inds.append(t * self.tiles_per_tiling + int(local))
        return inds

    def featurize(self, x: Iterable[float]) -> np.ndarray:
        v = np.zeros(self.total_tiles, dtype=float)
        for i in self.active_indices(x):
            v[i] = 1.0
        return v

class ActionBlockTileCoder:
    def __init__(self, tilecoder: TileCoder, n_actions: int):
        self.tc = tilecoder
        self.nA = int(n_actions)
        self.d = self.tc.total_tiles * self.nA

    def phi(self, x, a: int) -> np.ndarray:
        v = np.zeros(self.d, dtype=float)
        inds = self.tc.active_indices(x)
        offset = a * self.tc.total_tiles
        for i in inds:
            v[offset + i] = 1.0
        return v

    def active_count(self) -> int:
        return self.tc.n_tilings
