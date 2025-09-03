﻿# Reinforcement Learning Explained â€” Companion Code

## Build Status
[![ch2](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch2.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch2.yml)
[![ch3](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch3.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch3.yml)
[![ch4](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch4.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch4.yml)
[![ch5](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch5.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch5.yml)

---

This repository hosts **chapter-wise companion code** for the book *Reinforcement Learning Explained: From Theory to Practice*.  
It provides clean, minimal, and well-tested implementations of key reinforcement learning concepts.

---

## ðŸ“‚ Chapter Navigation
- [Chapter 2: The RL Problem Formulation](./ch2_rl_formulation)
- [Chapter 3: Multi-Armed Bandits](./ch3_multi_armed_bandits)
- [Chapter 4: Dynamic Programming Approaches](./ch4_dynamic_programming)
- [Chapter 5: Monte Carlo Methods](./ch5_monte_carlo)


---

## ðŸ“Š Chapter Progress

| Chapter | Title                          | Status        | Notes                                               |
|---------|--------------------------------|---------------|-----------------------------------------------------|
| 1       | Introduction                   | âœ”ï¸ Complete   | Book only (no code needed)                          |
| 2       | The RL Problem Formulation     | âœ”ï¸ Complete   | GridWorld, evaluation, policies, examples           |
| 3       | Multi-Armed Bandits            | âœ”ï¸ Complete   | Bandit envs, Îµ-greedy, UCB, Thompson                |
| 4       | Dynamic Programming Approaches | âœ”ï¸ Complete   | Policy Iteration, Value Iteration                   |
| 5       | Monte Carlo Methods            | â³ In Progress| Prediction, Control, On/Off-Policy done; refining   |
| 6+      | Temporal-Difference & Beyond   | âŒ Not Yet    | To be implemented in upcoming chapters              |

---

## ðŸ“‚ Repository Structure

```
rl-fundamentals-code/
â”œâ”€ ch2_rl_formulation/              # Chapter 2: The RL Problem Formulation
â”‚  â”œâ”€ gridworld.py                  # 4x4 GridWorld MDP (tabular P,R builder)
â”‚  â”œâ”€ evaluation.py                 # Policy evaluation, q_from_v(), greedy_from_q()
â”‚  â”œâ”€ policies.py                   # Deterministic & Îµ-greedy policies
â”‚  â”œâ”€ value_iteration.py            # Bellman optimality, value iteration
â”‚  â”œâ”€ examples/                     # Numeric examples, GridWorld demo, plotting
â”‚  â””â”€ tests/                        # Pytest-based checks for chapter numbers
â”‚
â”œâ”€ ch3_multi_armed_bandits/         # Chapter 3: Multi-Armed Bandits
â”‚  â”œâ”€ bandits.py                    # Bernoulli & Gaussian bandit environments
â”‚  â”œâ”€ epsilon_greedy.py             # Sample-average Îµ-greedy agent
â”‚  â”œâ”€ ucb.py                        # UCB1 agent
â”‚  â”œâ”€ thompson.py                   # Thompson Sampling (Betaâ€“Bernoulli)
â”‚  â”œâ”€ experiments.py                # Run algorithms, generate regret plots
â”‚  â”œâ”€ plots/                        # Saved figures
â”‚  â””â”€ tests/                        # Regression tests (ordering, sublinear regret)
â”‚
â”œâ”€ ch4_dynamic_programming/         # Chapter 4: Dynamic Programming Approaches
â”‚  â”œâ”€ gridworld.py                  # 4x4 deterministic GridWorld
â”‚  â”œâ”€ policy_evaluation.py          # Iterative policy evaluation
â”‚  â”œâ”€ policy_iteration.py           # Howardâ€™s policy iteration
â”‚  â”œâ”€ value_iteration.py            # Bellman optimality (value iteration)
â”‚  â”œâ”€ utils.py                      # Random + greedy helpers
â”‚  â”œâ”€ examples/                     # Run PI/VI demos
â”‚  â””â”€ tests/                        # Pytest checks for DP convergence
â”‚
â”œâ”€ ch5_monte_carlo/                 # Chapter 5: Monte Carlo Methods
â”‚  â”œâ”€ examples/
â”‚  â”‚   â”œâ”€ mc_prediction_demo.py           # First-visit vs every-visit MC (two-state MDP)
â”‚  â”‚   â”œâ”€ mc_control_es_gridworld.py      # MC control with Exploring Starts
â”‚  â”‚   â”œâ”€ mc_control_onpolicy_gridworld.py# On-policy MC control with Îµ-soft policies
â”‚  â”‚   â””â”€ mc_offpolicy_is_demo.py         # Off-policy IS: ordinary vs weighted
â”‚  â””â”€ tests/
â”‚      â”œâ”€ test_mc_control.py              # GridWorld control tests (MC-ES & on-policy)
â”‚      â””â”€ test_offpolicy_is.py            # Off-policy IS variance checks
â”‚
â”œâ”€ utils/                           # Shared helper utilities (future use)
â”œâ”€ .github/workflows/               # CI: runs pytest on every push/PR
â”‚  â””â”€ python-tests.yml
â”œâ”€ requirements.txt                 # Global dependencies
â””â”€ README.md                        # Project overview + usage
```

---

## ðŸš€ Getting Started

Clone this repository:

```bash
git clone https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code.git
cd Reinforcement-Learning-Explained-Code
```

Set up a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scripts\Activate    # Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## âœ… Running Tests

To run all tests:

```bash
python -m pytest -q
```

Run only Chapter 2 tests:

```bash
python -m pytest -q ch2_rl_formulation/tests
```

Run only Chapter 3 tests:

```bash
python -m pytest -q ch3_multi_armed_bandits/tests
```

Run only Chapter 4 tests:

```bash
python -m pytest -q ch4_dynamic_programming/tests
```

Run only Chapter 5 tests:

```bash
python -m pytest -q ch5_monte_carlo/tests
```

---

## ðŸ§ª Examples

Run numeric checks for Chapter 2:

```bash
python -m ch2_rl_formulation.examples.numeric_checks
```

Run the GridWorld demo (Chapter 2):

```bash
python -m ch2_rl_formulation.examples.gridworld_demo
```

Run Policy Iteration demo (Chapter 4):

```bash
python -m ch4_dynamic_programming.examples.run_policy_iteration
```

Run Value Iteration demo (Chapter 4):

```bash
python -m ch4_dynamic_programming.examples.run_value_iteration
```

Run MC prediction demo (Chapter 5):

```bash
python -m ch5_monte_carlo.examples.mc_prediction_demo
```

Run MC control with Exploring Starts (Chapter 5):

```bash
python -m ch5_monte_carlo.examples.mc_control_es_gridworld
```

Run on-policy MC control with Îµ-soft policies (Chapter 5):

```bash
python -m ch5_monte_carlo.examples.mc_control_onpolicy_gridworld
```

Run off-policy IS demo (Chapter 5):

```bash
python -m ch5_monte_carlo.examples.mc_offpolicy_is_demo
```

---

## âš™ï¸ Continuous Integration

- GitHub Actions (`.github/workflows/python-tests.yml`) automatically run tests for all chapters on every push and pull request.
- This ensures correctness and reproducibility of the examples.

---

## ðŸ“š How to Cite

If you use this code or the accompanying book in your research or teaching, please cite:

**Book (forthcoming):**
```bibtex
@book{baride2025rlfundamentals,
  author    = {Srikanth Baride},
  title     = {Reinforcement Learning Explained},
  publisher = {yet to decide},
  year      = {2025},
  note      = {In preparation}
}
```

**Companion Code (GitHub):**
```bibtex
@misc{baride2025rlcode,
  author       = {Srikanth Baride},
  title        = {Reinforcement Learning Explained â€” Companion Code},
  year         = {2025},
  howpublished = {\url{https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code}},
  note         = {Accessed: YYYY-MM-DD}
}
```

---

## ðŸ¤ Contributing

Contributions are welcome! To contribute:

1. **Fork the repository** and create your branch:  
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**, following the existing folder structure and style:  
   - Keep chapter code inside its respective `chX_*` folder.  
   - Use clear function names and docstrings.  
   - Write minimal, didactic code (prioritize readability over optimization).  
   - Add tests in the corresponding `tests/` folder.  

3. **Run tests locally** before submitting:  
   ```bash
   python -m pytest -q
   ```

4. **Commit and push** your changes:  
   ```bash
   git commit -m "Add feature: description"
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub.  

For issues, please use the GitHub **Issues tab** and provide:  
- A clear description of the problem.  
- Steps to reproduce (if itâ€™s a bug).  
- Suggested fix or clarification request (if possible).  

---

## ðŸ“– License

MIT License Â© 2025 â€” Srikanth Baride

