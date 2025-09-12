import numpy as np
from ..envs.bandit import TwoArmedBandit
from ..policies.softmax import SoftmaxPolicy
from ..agents.reinforce import Reinforce

def run(episodes=200, seed=0):
    env = TwoArmedBandit(q_star=(1.0, 1.5), seed=seed)
    x = np.array([1.0], dtype=float)
    policy = SoftmaxPolicy(nA=2, d=1, seed=seed)
    algo = Reinforce(gamma=1.0, alpha=0.05, normalize_adv=True, baseline_fn=None, seed=seed)
    probs_hist = []
    class EPEnv:
        def reset(self): return x
        def step(self, a):
            _, r, done, _ = env.step(a)
            return None, r, True, {}
    for _ in range(episodes):
        traj = algo.run_episode_discrete(EPEnv(), policy, lambda s: s)
        algo.update_discrete(traj, policy, lambda s: s)
        probs_hist.append(policy.probs(x).copy())
    return np.array(probs_hist), policy.theta

if __name__ == "__main__":
    probs, theta = run()
    print("Final action probabilities:", probs[-1])
