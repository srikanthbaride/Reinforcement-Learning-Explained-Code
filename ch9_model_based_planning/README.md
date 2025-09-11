# Chapter 9 — Model-Based RL & Planning (Dyna)

This folder follows the repository's chapter-wise layout and provides minimal, didactic code to accompany the chapter *Model-Based RL and Planning*.

## Contents
- `gridworld.py` — Deterministic n-by-n grid (terminal goal, step cost).
- `tabular_model.py` — Count-based tabular model (learns \hat{P}, \hat{R}).
- `q_learning.py` — Plain Q-learning (baseline).
- `dyna_q.py` — Dyna-Q agent (Q-learning + planning with budget k).
- `rollout_planner.py` — Short-rollout planning on the learned model.
- `examples/chain_example.py` — Two-step chain reproducing Dyna-Q speedup table.
- `examples/run_dyna_grid.py` — 5-by-5 gridworld, compare k in {0,5,10,20} (saves plots).
- `examples/rollout_vs_q.py` — 2-by-2 grid: Pure Q-learning vs short rollouts.
- `tests/test_ch9_mbrl.py` — Smoke tests to validate key numbers/behaviors.

## Quick Start
```bash
conda create -n rl-ch9-mbrl python=3.9 -y
conda activate rl-ch9-mbrl
pip install numpy matplotlib

# 1) Two-step chain (numeric table)
python ch9_model_based_planning/examples/chain_example.py

# 2) Dyna-Q on 5x5 (saves plots in ch9_model_based_planning/outputs/)
python ch9_model_based_planning/examples/run_dyna_grid.py

# 3) 2x2: model-free vs short-rollout planning (saves plots)
python ch9_model_based_planning/examples/rollout_vs_q.py
```
All figures use Matplotlib only, single plot per figure, with default colors.
