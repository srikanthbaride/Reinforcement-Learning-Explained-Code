# ch8_td_lambda · Eligibility Traces and TD(λ)

Reference implementations and experiments for **Chapter 8** of  
_Reinforcement Learning Fundamentals: From Theory to Practice_.

This chapter covers the forward/backward views, TD(λ) prediction, SARSA(λ) control, and true-online TD(λ) with linear function approximation.

---

## Folder layout

```
ch8_td_lambda/
├─ gridworld_small.py              # 4×4 tabular gridworld (start=(3,0), goal=(0,3))
├─ td_lambda.py                    # TD(λ) prediction (backward view; accumulating/replacing)
├─ sarsa_lambda.py                 # SARSA(λ) control with ε-greedy
├─ true_online_td_lambda.py        # True Online TD(λ) for linear FA
├─ plot_tdlambda_learning.py       # Produces learning curves for λ ∈ {0, 0.5, 1}
├─ tests/
│  └─ test_forward_backward_equiv.py  # Forward ↔ backward numerical check
```

---

## Quick start

> Assumes Python ≥ 3.9 and `matplotlib`, `numpy`, `pytest`.

### 1) Run the unit test (forward ↔ backward equivalence)

```bash
pytest ch8_td_lambda/tests -q
```

Expected:
```
.                                                                   [100%]
1 passed in ~0.02s
```

### 2) Generate learning curves (SARSA(λ) on gridworld)

```bash
python ch8_td_lambda/plot_tdlambda_learning.py
```

Artifacts written to the project (figure under `figs/`):
- `ch8_tdlambda_learning.csv`
- `figs/ch8_tdlambda_learning.png`

The plot compares success rates for **λ ∈ {0.0, 0.5, 1.0}**.  
(Intermediate λ typically balances speed and stability in this task.)

---

## Minimal examples

### TD(λ) prediction (tabular; backward view)
```python
import numpy as np
from ch8_td_lambda.gridworld_small import GridworldSmall
from ch8_td_lambda.td_lambda import td_lambda_prediction

env = GridworldSmall(seed=0)

def random_policy(s: int):
    return np.ones(env.n_actions) / env.n_actions  # uniform

V = td_lambda_prediction(env, random_policy, gamma=0.99, alpha=0.1, lam=0.9, episodes=200)
print(V.reshape(env.n_rows, env.n_cols))
```

### SARSA(λ) control (ε-greedy)
```python
from ch8_td_lambda.gridworld_small import GridworldSmall
from ch8_td_lambda.sarsa_lambda import sarsa_lambda_control
import numpy as np

env = GridworldSmall(seed=0)
Q = sarsa_lambda_control(env, gamma=0.99, alpha=0.1, lam=0.8, epsilon=0.1, episodes=1000)
print(Q.argmax(axis=1).reshape(env.n_rows, env.n_cols))  # greedy policy
```

### True Online TD(λ) (linear FA; one-hot features)
```python
import numpy as np
from ch8_td_lambda.gridworld_small import GridworldSmall
from ch8_td_lambda.true_online_td_lambda import true_online_td_lambda_linear

env = GridworldSmall(seed=0)
def phi(s: int):
    x = np.zeros(env.n_states, dtype=float)
    x[s] = 1.0
    return x

w = true_online_td_lambda_linear(env, phi, gamma=0.99, alpha=0.15, lam=0.8, episodes=800, seed=0)
print(w.reshape(env.n_rows, env.n_cols))  # value estimates
```

---

## Expected outputs

- **Learning curves:** `ch8_tdlambda_learning.png` — success rate vs. episodes for λ=0, 0.5, 1.0.  
- **CSV:** `ch8_tdlambda_learning.csv` — columns: `episodes, lambda_0.0, lambda_0.5, lambda_1.0`.

---

## LaTeX snippet (embed figure in the book)

After generating the figure, move/commit it under `figs/` and include:

```latex
\begin{figure}[h!]
  \centering
  \includegraphics[width=0.75\linewidth]{figs/ch8_tdlambda_learning.png}
  \caption{Learning curves for TD($\lambda$) on a $4\times4$ gridworld under SARSA($\lambda$). Intermediate $\lambda$ values (e.g., $0.5$) often balance speed and stability.}
  \label{fig:tdlambda-learning}
\end{figure}
```

---

## Notes

- `sarsa_lambda_control` supports `trace_type="accumulating"` or `"replacing"` (default is replacing in the learning-curve script for stability when states repeat).
- For reproducibility, seeds are set inside scripts; you can adjust α, ε, and λ from the script/CLI if desired.

---

## References

- Sutton, R. S. (1988). *Learning to Predict by the Methods of Temporal Differences*.  
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd ed.)*.  
- van Seijen, H., & Sutton, R. S. (2014). *True Online TD(λ)*.  
- Tesauro, G. (1995). *TD-Gammon*.  
- Schulman, J. et al. (2016). *Generalized Advantage Estimation*.
