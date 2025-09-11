import numpy as np

class TabularModel:
    """
    Count-based one-step tabular model.
    Stores transition counts and average rewards; predicts using empirical means.
    """
    def __init__(self, S, A):
        self.S, self.A = S, A
        self.counts = np.zeros((S, A, S), dtype=float)
        self.rew_sum = np.zeros((S, A), dtype=float)
        self.rew_count = np.zeros((S, A), dtype=float)
        self.seen = set()

    def update(self, s, a, r, s2):
        self.counts[s, a, s2] += 1.0
        self.rew_sum[s, a] += r
        self.rew_count[s, a] += 1.0
        self.seen.add((s, a))

    def predict(self, s, a):
        if (s, a) not in self.seen:
            return s, 0.0
        probs = self.counts[s, a] / max(1.0, self.counts[s, a].sum())
        s2 = np.random.choice(self.S, p=probs)
        r = self.rew_sum[s, a] / max(1.0, self.rew_count[s, a])
        return int(s2), float(r)
