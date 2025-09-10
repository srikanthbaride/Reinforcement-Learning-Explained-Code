from __future__ import annotations
import numpy as np
from ch8_td_lambda.gridworld_small import GridworldSmall
from ch8_td_lambda.td_lambda import td_lambda_prediction

def main():
    env = GridworldSmall(seed=0)

    def random_policy(s: int):
        return np.ones(env.n_actions) / env.n_actions  # uniform

    V = td_lambda_prediction(env, random_policy, gamma=0.99, alpha=0.1, lam=0.9, episodes=300, seed=0)
    print('Value estimates (4x4):\n', V.reshape(env.n_rows, env.n_cols))

if __name__ == '__main__':
    main()
