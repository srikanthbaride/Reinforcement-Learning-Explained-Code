# ch7_td_control/run_cliff.py
from __future__ import annotations
import argparse
from .cliff_env import CliffWalkingEnv
from .policies import fixed_epsilon, linear_decay, exp_decay, inverse_glie
from .sarsa import sarsa
from .q_learning import q_learning
from .plot_utils import plot_learning_curves

def make_schedule(name: str, **kwargs):
    name = name.lower()
    if name == "fixed": return fixed_epsilon(kwargs.get("eps",0.1))
    if name == "linear": return linear_decay(kwargs.get("eps0",0.3), kwargs.get("eps_min",0.05), kwargs.get("T",500))
    if name == "exp": return exp_decay(kwargs.get("eps0",0.3), kwargs.get("k",0.01), kwargs.get("eps_min",0.0))
    if name == "glie": return inverse_glie(kwargs.get("c",1.0))
    raise ValueError(f"Unknown schedule {name}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schedule", type=str, default="fixed", choices=["fixed","linear","exp","glie"])
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--eps_min", type=float, default=0.05)
    p.add_argument("--linear_T", type=int, default=500)
    p.add_argument("--exp_k", type=float, default=0.01)
    p.add_argument("--glie_c", type=float, default=1.0)
    p.add_argument("--ma_k", type=int, default=20)
    args = p.parse_args()

    env = CliffWalkingEnv(seed=args.seed)
    if args.schedule=="fixed": sched = make_schedule("fixed", eps=args.eps)
    elif args.schedule=="linear": sched = make_schedule("linear", eps0=args.eps, eps_min=args.eps_min, T=args.linear_T)
    elif args.schedule=="exp": sched = make_schedule("exp", eps0=args.eps, k=args.exp_k, eps_min=args.eps_min)
    else: sched = make_schedule("glie", c=args.glie_c)

    sarsa_Q, sarsa_returns = sarsa(env, episodes=args.episodes, alpha=args.alpha, gamma=args.gamma, eps_schedule=sched, seed=args.seed)
    ql_Q, ql_returns = q_learning(env, episodes=args.episodes, alpha=args.alpha, gamma=args.gamma, eps_schedule=sched, seed=args.seed)
    episodes = list(range(args.episodes))
    plot_learning_curves(episodes, sarsa_returns, ql_returns, ma_k=args.ma_k)

if __name__ == "__main__":
    main()
