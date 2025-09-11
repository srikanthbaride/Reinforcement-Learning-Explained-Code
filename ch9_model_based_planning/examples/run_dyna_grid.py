import os, json
import numpy as np
from ch9_model_based_planning.gridworld import GridWorld
from ch9_model_based_planning.tabular_model import TabularModel
from ch9_model_based_planning.dyna_q import DynaQ
from ch9_model_based_planning.q_learning import QLearning
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

def run_episode(env, agent, max_steps=500):
    s = env.reset()
    G = 0.0
    for _ in range(max_steps):
        a = agent.act(s)
        s2, r, done, _ = env.step(a)
        if hasattr(agent, 'update_real'):
            agent.update_real(s, a, r, s2, done)
            agent.plan()
        else:
            agent.update(s, a, r, s2, done)
        s = s2
        G += r
        if done:
            break
    return G

def experiment(n_episodes=300, planning_values=(0,5,10,20)):
    env = GridWorld(n=5, step_cost=-1.0, goal=(0,4), start=(4,0), gamma=1.0)
    curves = {}
    for k in planning_values:
        if k == 0:
            agent = QLearning(env.S, 4, alpha=0.5, gamma=1.0, epsilon=0.1)
        else:
            model = TabularModel(env.S, 4)
            agent = DynaQ(env.S, 4, model, alpha=0.5, gamma=1.0, epsilon=0.1, planning_steps=k)
        returns = [run_episode(env, agent) for _ in range(n_episodes)]
        curves[k] = returns
        os.makedirs(OUTDIR, exist_ok=True)
        plt.figure()
        plt.plot(returns)
        plt.title(f'Dyna-Q (k={k}) vs episodes')
        plt.xlabel('Episode'); plt.ylabel('Return per episode')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'dyna_k{k}_returns.png'), dpi=150)
        plt.close()
    with open(os.path.join(OUTDIR, 'dyna_summary.json'), 'w') as f:
        import json
        f.write(json.dumps({str(k): v for k, v in curves.items()}))
    return curves

if __name__ == '__main__':
    curves = experiment()
    for k, arr in curves.items():
        print(f'k={k}: last-10 avg = {np.mean(arr[-10:]):.2f}')
