import numpy as np
from ch11_policy_gradient.envs.bandit import TwoArmedBandit
from ch11_policy_gradient.policies.softmax import SoftmaxPolicy
from ch11_policy_gradient.agents.reinforce import Reinforce

def test_softmax_probs_increase_for_better_arm():
    seed = 123
    env = TwoArmedBandit(q_star=(1.0, 1.5), seed=seed)
    x = np.array([1.0], dtype=float)
    policy = SoftmaxPolicy(nA=2, d=1, seed=seed)
    algo = Reinforce(gamma=1.0, alpha=0.05, normalize_adv=True, baseline_fn=None, seed=seed)

    class EPEnv:
        def reset(self): return x
        def step(self, a):
            _, r, done, _ = env.step(a)
            return None, r, True, {}

    p0 = policy.probs(x)[1]
    for _ in range(60):
        traj = algo.run_episode_discrete(EPEnv(), policy, lambda s: s)
        algo.update_discrete(traj, policy, lambda s: s)
    p1 = policy.probs(x)[1]
    assert p1 >= p0 - 1e-6
    assert p1 > 0.55

def test_logprob_gradient_shape_and_finiteness():
    x = np.array([1.0])
    policy = SoftmaxPolicy(nA=3, d=1, seed=0)
    logp, grad = policy.logprob_and_grad(x, a=2)
    assert grad.shape == (3,1)
    assert np.isfinite(logp)
