from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ch8_td_lambda.plot_utils import moving_average, style_axes
from ch8_td_lambda.gridworld_small import GridworldSmall
from ch8_td_lambda.sarsa_lambda import sarsa_lambda_control

def eval_success_rate(env, Q, episodes=200, max_steps=200, seed=123) -> float:
    rng = np.random.default_rng(seed)
    succ = 0
    for _ in range(episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = int(np.argmax(Q[s]))
            s, r, done, *_ = env.step(a)
            if done:
                succ += 1
                break
    return succ / episodes

def main():
    seeds = [0, 1, 2]
    lambdas = [0.0, 0.5, 1.0]
    episodes_per_seed = 3000
    eval_every = 100
    alphas = {0.0: 0.15, 0.5: 0.12, 1.0: 0.08}  # gentle tuning

    curves = {lam: [] for lam in lambdas}
    xs = []

    for lam in lambdas:
        agg = []
        for sd in seeds:
            env = GridworldSmall(seed=sd)
            # train in chunks, evaluate periodically
            Q = np.zeros((env.n_states, env.n_actions))
            for ep0 in range(0, episodes_per_seed, eval_every):
                Q = sarsa_lambda_control(
                    env=env, gamma=0.99, alpha=alphas[lam], lam=lam,
                    epsilon=0.1, episodes=eval_every, trace_type='replacing',
                    seed=sd, n_states=env.n_states, n_actions=env.n_actions
                )
                sr = eval_success_rate(env, Q, episodes=100, seed=sd)
                agg.append(sr)
                if lam == lambdas[0]:
                    xs.append(ep0 + eval_every)
        curves[lam] = np.array(agg).reshape(len(seeds), -1).mean(axis=0)

    # plot
    plt.figure(figsize=(7.2, 4.2))
    for lam in lambdas:
        plt.plot(xs, moving_average(curves[lam], w=5), label=f'λ={lam}')
    style_axes(plt.gca(), xlabel='Episodes', ylabel='Success rate (greedy)', ylim=(0,1.02), legend_loc='lower right')
    import os
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/ch8_tdlambda_learning.png', dpi=160)

    # also dump CSV
    import csv
    with open('ch8_tdlambda_learning.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episodes'] + [f'lambda_{lam}' for lam in lambdas])
        for i, x in enumerate(xs):
            row = [x] + [float(curves[lam][i]) for lam in lambdas]
            w.writerow(row)

if __name__ == '__main__':
    main()
