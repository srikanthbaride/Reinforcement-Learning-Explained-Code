import numpy as np

class DynaQ:
    """
    Dyna-Q agent: Q-learning + planning over a tabular model.
    Model must support update(s,a,r,s2) and predict(s,a).
    """
    def __init__(self, S, A, model, alpha=0.5, gamma=1.0, epsilon=0.1, planning_steps=10):
        self.S, self.A = S, A
        self.Q = np.zeros((S, A), dtype=float)
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = int(planning_steps)
        self.seen = set()

    def act(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.A)
        return int(np.argmax(self.Q[s]))

    def update_real(self, s, a, r, s2, done):
        target = r + (0.0 if done else self.gamma * np.max(self.Q[s2]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.model.update(s, a, r, s2)
        self.seen.add((s, a))

    def plan(self):
        if not self.seen:
            return
        seen_list = list(self.seen)
        for _ in range(self.planning_steps):
            s, a = seen_list[np.random.randint(len(seen_list))]
            s2, r = self.model.predict(s, a)
            target = r + self.gamma * np.max(self.Q[s2])
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
