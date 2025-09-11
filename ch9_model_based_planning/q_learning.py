import numpy as np

class QLearning:
    def __init__(self, S, A, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.S, self.A = S, A
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((S, A), dtype=float)

    def act(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.A)
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s2, done):
        target = r + (0.0 if done else self.gamma * np.max(self.Q[s2]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
