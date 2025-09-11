import os
import numpy as np
import matplotlib.pyplot as plt
from ch9_model_based_planning.gridworld import GridWorld
from ch9_model_based_planning.tabular_model import TabularModel
from ch9_model_based_planning.q_learning import QLearning
from ch9_model_based_planning.rollout_planner import rollout_q_update

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

def run_q_learning(env, n_episodes=200, alpha=0.5, epsilon=0.1):
    agent = QLearning(env.S, 4, alpha=alpha, gamma=env.gamma, epsilon=epsilon)
    returns = []
    for ep in range(n_episodes):
        s = env.reset()
        G, steps = 0.0, 0
        while True:
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
            G += r
            steps += 1
            if done or steps > 500:
                break
        returns.append(G)
    return returns, agent.Q

def run_rollout_planning(env, n_episodes=200, alpha=0.5, epsilon=0.1, rollout_len=1, rollouts_per_step=1):
    agent = QLearning(env.S, 4, alpha=alpha, gamma=env.gamma, epsilon=epsilon)
    model = TabularModel(env.S, 4)
    returns = []
    for ep in range(n_episodes):
        s = env.reset()
        G, steps = 0.0, 0
        while True:
            a = agent.act(s)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            model.update(s, a, r, s2)
            for _ in range(rollouts_per_step):
                a0 = int(np.argmax(agent.Q[s]))
                a_seq = [a0] + [np.random.randint(4) for _ in range(rollout_len-1)]
                rollout_q_update(agent.Q, model, s, a_seq, alpha=alpha, gamma=env.gamma)
            s = s2
            G += r
            steps += 1
            if done or steps > 500:
                break
        returns.append(G)
    return returns, agent.Q

if __name__ == '__main__':
    env = GridWorld(n=2, step_cost=0.0, goal=(0,1), start=(1,0), gamma=1.0)
    q_ret, _ = run_q_learning(env, n_episodes=100)
    ro_ret, _ = run_rollout_planning(env, n_episodes=100, rollout_len=2, rollouts_per_step=1)
    os.makedirs(OUTDIR, exist_ok=True)
    plt.figure(); plt.plot(q_ret); plt.title('Pure Q-learning (2x2)'); plt.xlabel('Episode'); plt.ylabel('Return'); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, 'rollout_vs_q_qlearn.png'), dpi=150); plt.close()
    plt.figure(); plt.plot(ro_ret); plt.title('Short-rollout planning (2x2)'); plt.xlabel('Episode'); plt.ylabel('Return'); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, 'rollout_vs_q_rollout.png'), dpi=150); plt.close()
    print(f'Final-10 avg: Q-learning={np.mean(q_ret[-10:]):.2f}, Rollout={np.mean(ro_ret[-10:]):.2f}')
