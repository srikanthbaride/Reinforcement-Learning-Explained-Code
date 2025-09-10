from __future__ import annotations
import numpy as np
from ch8_td_lambda.gridworld_small import GridworldSmall
from ch8_td_lambda.sarsa_lambda import sarsa_lambda_control

def main():
    env = GridworldSmall(seed=0)
    Q = sarsa_lambda_control(env, gamma=0.99, alpha=0.1, lam=0.8, epsilon=0.1, episodes=1500, seed=0)
    greedy = Q.argmax(axis=1).reshape(env.n_rows, env.n_cols)
    print('Greedy policy (0:up,1:right,2:down,3:left):\n', greedy)

if __name__ == '__main__':
    main()
