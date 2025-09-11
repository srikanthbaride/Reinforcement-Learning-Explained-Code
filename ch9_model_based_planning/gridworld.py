import numpy as np

ACTIONS = [(0,1), (0,-1), (-1,0), (1,0)]  # R, L, U, D

class GridWorld:
    """
    Deterministic n-by-n gridworld.
    - step_cost: reward on non-terminal transitions (e.g., -1 or 0)
    - goal: (row, col) terminal with 0 reward upon entering
    - start: starting (row, col)
    - gamma: discount
    On wall hit, agent stays in place.
    """
    def __init__(self, n=5, step_cost=-1.0, goal=(0,4), start=(4,0), gamma=1.0):
        self.n = int(n)
        self.S = n*n
        self.A = 4
        self.step_cost = float(step_cost)
        self.goal = tuple(goal)
        self.start = tuple(start)
        self.gamma = float(gamma)
        self.state = self._to_idx(self.start)

    def _to_idx(self, rc):
        r, c = rc
        return r*self.n + c

    def _to_rc(self, s):
        return divmod(s, self.n)

    def reset(self):
        self.state = self._to_idx(self.start)
        return self.state

    def step(self, a):
        assert 0 <= a < self.A
        r, c = self._to_rc(self.state)
        if (r, c) == self.goal:
            return self.state, 0.0, True, {}
        dr, dc = ACTIONS[a]
        nr, nc = r+dr, c+dc
        if nr < 0 or nr >= self.n or nc < 0 or nc >= self.n:
            nr, nc = r, c
        s2 = self._to_idx((nr, nc))
        rwd = 0.0 if (nr, nc) == self.goal else self.step_cost
        done = (nr, nc) == self.goal
        self.state = s2
        return s2, rwd, done, {}

    def transitions(self, s, a):
        r, c = self._to_rc(s)
        if (r, c) == self.goal:
            return [(s, 1.0, 0.0)]
        dr, dc = ACTIONS[a]
        nr, nc = r+dr, c+dc
        if nr < 0 or nr >= self.n or nc < 0 or nc >= self.n:
            nr, nc = r, c
        s2 = self._to_idx((nr, nc))
        rwd = 0.0 if (nr, nc) == self.goal else self.step_cost
        return [(s2, 1.0, rwd)]
