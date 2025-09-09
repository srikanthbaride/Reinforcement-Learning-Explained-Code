# ch7_td_control/cliff_env.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Tuple, List

Action = int
State = Tuple[int, int]

@dataclass
class DiscreteSpace:
    n: int
    def sample(self) -> int:
        return random.randrange(self.n)

class CliffWalkingEnv:
    def __init__(self, rows: int = 4, cols: int = 12, seed: int | None = 0):
        self.rows, self.cols = rows, cols
        self.start = (rows - 1, 0)
        self.goal  = (rows - 1, cols - 1)
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}
        self._state: State = self.start
        self.action_space = DiscreteSpace(4)
        self.state_space_n = rows * cols
        if seed is not None:
            random.seed(seed)

    def reset(self) -> State:
        self._state = self.start
        return self._state

    def step(self, a: Action):
        r, c = self._state
        nr, nc = self._move(r, c, a)
        reward = -1.0
        done = False
        if (nr, nc) in self.cliff:
            reward = -100.0
            self._state = self.start
            done = False
            return self._state, reward, done, {}
        if (nr, nc) == self.goal:
            self._state = (nr, nc)
            done = True
            return self._state, reward, done, {}
        self._state = (nr, nc)
        return self._state, reward, done, {}

    def _move(self, r: int, c: int, a: Action) -> State:
        if a == 0: r = max(0, r - 1)
        elif a == 1: c = min(self.cols - 1, c + 1)
        elif a == 2: r = min(self.rows - 1, r + 1)
        elif a == 3: c = max(0, c - 1)
        else: raise ValueError(f"Invalid action {a}")
        return (r, c)

    def render(self, path: List[State] | None = None) -> None:
        path = set(path or [])
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                s = (r, c)
                if s == self._state: ch = 'A'
                elif s == self.start: ch = 'S'
                elif s == self.goal:  ch = 'G'
                elif s in self.cliff: ch = 'X'
                else: ch = '.'
                if s in path and ch not in {'S','G'}:
                    ch = '*'
                row.append(ch)
            print(' '.join(row))
        print()
