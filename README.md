# Reinforcement Learning Explained — Companion Code

## Build Status
[![ch2](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch2.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch2.yml)
[![ch3](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch3.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch3.yml)
[![ch4](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch4.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch4.yml)
[![ch5](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch5.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch5.yml)
[![ch6](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch6.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch6.yml)
[![ch7](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch7.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch7.yml)
[![ch8](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch8.yml/badge.svg)](https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code/actions/workflows/ch8.yml)

---

This repository hosts **chapter-wise companion code** for the book *Reinforcement Learning Explained*.  
It provides clean, minimal, and well-tested implementations of key reinforcement learning concepts.

---

## 📑 Chapter Navigation
- [Chapter 2: The RL Problem Formulation](./ch2_rl_formulation)
- [Chapter 3: Multi-Armed Bandits](./ch3_multi_armed_bandits)
- [Chapter 4: Dynamic Programming Approaches](./ch4_dynamic_programming)
- [Chapter 5: Monte Carlo Methods](./ch5_monte_carlo)
- [Chapter 6: Temporal-Difference Learning](./ch6_td_learning)
- [Chapter 7: TD Control — SARSA and Q-Learning](./ch7_td_control)
- [Chapter 8: Eligibility Traces and TD(λ)](./ch8_td_lambda)

---

## 📊 Chapter Progress

| Chapter | Title                          | Status        | Notes                                               |
|---------|--------------------------------|---------------|-----------------------------------------------------|
| 1       | Introduction                   | ✅ Complete    | Book only (no code needed)                          |
| 2       | The RL Problem Formulation     | ✅ Complete    | GridWorld, evaluation, policies, examples           |
| 3       | Multi-Armed Bandits            | ✅ Complete    | Bandit envs, ε-greedy, UCB, Thompson                |
| 4       | Dynamic Programming Approaches | ✅ Complete    | Policy Iteration, Value Iteration                   |
| 5       | Monte Carlo Methods            | ✅ Complete    | Prediction, Control, On/Off-Policy                  |
| 6       | Temporal-Difference Learning   | ✅ Complete    | TD(0), n-step TD, prediction examples               |
| 7       | TD Control                     | ✅ Complete    | SARSA, Q-learning, Cliff-Walking, exploration       |
| 8       | Eligibility Traces and TD(λ)   | ✅ Complete    | TD(λ), SARSA(λ), True Online TD(λ), gridworld demos |

---

## 📂 Repository Structure

```
rl-fundamentals-code/
├─ ch2_rl_formulation/             # Chapter 2
├─ ch3_multi_armed_bandits/        # Chapter 3
├─ ch4_dynamic_programming/        # Chapter 4
├─ ch5_monte_carlo/                # Chapter 5
├─ ch6_td_learning/                # Chapter 6
├─ ch7_td_control/                 # Chapter 7
├─ ch8_td_lambda/                  # Chapter 8
├─ utils/
└─ .github/workflows/
```

---

## ✅ Running Tests

To run all tests:

```bash
python -m pytest -q
```

Run only Chapter 8 tests:

```bash
python -m pytest -q ch8_td_lambda/tests
```

---

## ⚙️ Continuous Integration

- GitHub Actions (`.github/workflows/*.yml`) automatically run tests for each chapter on every push and pull request.
- This ensures correctness and reproducibility of the examples.

---

## 📚 How to Cite

If you use this code or the accompanying book in your research or teaching, please cite:

**Book (forthcoming):**
```bibtex
@book{baride2025rlexplained,
  author    = {Srikanth Baride et al. },
  title     = {Reinforcement Learning Explained},
  publisher = {To be decided},
  year      = {2025},
  note      = {In preparation}
}
```

**Companion Code (GitHub):**
```bibtex
@misc{baride2025rlcode,
  author       = {Srikanth Baride},
  title        = {Reinforcement Learning Explained — Companion Code},
  year         = {2025},
  howpublished = {\url{https://github.com/srikanthbaride/Reinforcement-Learning-Explained-Code}},
  note         = {Accessed: YYYY-MM-DD}
}
```

---

## 📖 License

MIT License © 2025 — Srikanth Baride
