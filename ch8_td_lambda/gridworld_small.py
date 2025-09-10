# Minimal 4x4 gridworld with gym-like API (tabular states 0..15).
# Start at (3,0), goal at (0,3). Step reward -0.01, goal reward +1.
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

class GridworldSmall:
    def __init__(self, seed: Optional[int] = None):
        self.n_rows = 4
        self.n_cols = 4
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4  # 0:up, 1:right, 2:down, 3:left
        self.start = (3, 0)
        self.goal = (0, 3)
        self.step_reward = -0.01
        self.goal_reward = 1.0
        self._rng = np.random.default_rng(seed)
        self.s = self._to_state(self.start)

    def _to_state(self, rc: Tuple[int, int]) -> int:
        r, c = rc
        return r * self.n_cols + c

    def _to_rc(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.n_cols)

    def reset(self) -> int:
        self.s = self._to_state(self.start)
        return self.s

    def step(self, a: int):
        r, c = self._to_rc(self.s)
        if a == 0:   # up
            r = max(0, r - 1)
        elif a == 1: # right
            c = min(self.n_cols - 1, c + 1)
        elif a == 2: # down
            r = min(self.n_rows - 1, r + 1)
        elif a == 3: # left
            c = max(0, c - 1)
        s_next = self._to_state((r, c))
        done = (r, c) == self.goal
        reward = self.goal_reward if done else self.step_reward
        self.s = s_next
        return s_next, reward, done, {}

    def render(self) -> None:
        r, c = self._to_rc(self.s)
        board = np.full((self.n_rows, self.n_cols), '.', dtype=object)
        board[self.goal] = 'G'
        board[r, c] = 'A'
        print('\n'.join(' '.join(row) for row in board))
