# 01_Fundamentals

Foundational Reinforcement Learning algorithms implemented from scratch, following Sutton & Barto's *Reinforcement Learning: An Introduction*.

---

## 📁 Structure

```
01_Fundamentals/
├── code/
│   ├── src/
│   │   ├── GridWorld.py
│   │   ├── DP_policyeval.py
│   │   ├── DP_policy_iteration.py
│   │   ├── DP_valueiter.py
│   │   ├── MonteCarlo.py
│   │   └── TD0.py
│   ├── GridWorld.ipynb
│   ├── DP.ipynb
│   ├── MonteCarlo.ipynb
│   └── TD.ipynb
└── notes/
    ├── GridWorld.md
    ├── DynamicProgramming.md
    ├── MonteCarlo.md
    └── TemporalDifference.md
```

---

## 🌍 GridWorld

A custom GridWorld environment built from scratch with stochastic dynamics.

| File | Description |
|------|-------------|
| [`code/src/GridWorld.py`](code/src/GridWorld.py) | GridWorld environment class |
| [`code/GridWorld_test.ipynb`](code/GridWorld_test.ipynb) | Environment tests & visualization |
| [`notes/GridWorld.md`](notes/GridWorld.md) | Concepts & notes |

**Features:**
- Configurable size, obstacles, icy floors (stochastic transitions)
- `step()`, `reset()`, `render()`, `get_transition_prob()`, `get_dynamics_rewards()`
- Markov Property satisfied

---

## 🧮 Dynamic Programming

DP algorithms assuming full knowledge of environment dynamics.

| File | Description |
|------|-------------|
| [`code/src/DP_policyeval.py`](code/src/DP_policyeval.py) | Policy Evaluation |
| [`code/src/DP_policyiter.py`](code/src/DP_policyiter.py) | Policy Iteration |
| [`code/src/DP_valueiter.py`](code/src/DP_valueiter.py) | Value Iteration |
| [`code/Dynamic_Programming.ipynb`](code/Dynamic_Programming.ipynb) | DP experiments & comparisons |
| [`notes/DynamicProgramming.md`](notes/DynamicProgramming.md) | Concepts & notes |

**Algorithms:**
- Policy Evaluation
- Policy Iteration (Eval + Improvement loop)
- Value Iteration (1-step Eval + Improvement)

---

## 🎲 Monte Carlo

Model-free methods using sampled episodes to estimate value functions.

| File | Description |
|------|-------------|
| [`code/src/MonteCarlo.py`](code/src/MonteCarlo.py) | All MC methods |
| [`code/MonteCarlo.ipynb`](code/MonteCarlo.ipynb) | MC experiments & comparisons |
| [`notes/MonteCarlo.md`](notes/MonteCarlo.md) | Concepts & notes |

**Algorithms:**
- First-visit / Every-visit Prediction
- Exploring Starts Control
- On-policy MC Control (ε-greedy)
- Off-policy MC Prediction (Weighted Importance Sampling)

---

## ⏱️ Temporal Difference

Model-free methods that update at every step without waiting for episode end.

| File | Description |
|------|-------------|
| [`code/src/TD0.py`](code/src/TD0.py) | All TD methods |
| [`code/TemporalDifference.ipynb`](code/TemporalDifference.ipynb) | TD experiments & comparisons |
| [`notes/TemporalDifference.md`](notes/TemporalDifference.md) | Concepts & notes |

**Algorithms:**
- TD(0) Prediction
- SARSA (On-policy)
- Expected SARSA
- Q-learning (Off-policy)
- Double Q-learning

---

## 📊 Algorithm Comparison

| Algorithm | Model-free | Online | On/Off-policy |
|-----------|-----------|--------|---------------|
| Policy Evaluation | ❌ | ❌ | On |
| Policy Iteration | ❌ | ❌ | On |
| Value Iteration | ❌ | ❌ | On |
| MC Prediction | ✅ | ❌ | On |
| ES Control | ✅ | ❌ | On |
| On-policy MC | ✅ | ❌ | On |
| Off-policy MC | ✅ | ❌ | Off |
| TD(0) | ✅ | ✅ | On |
| SARSA | ✅ | ✅ | On |
| Expected SARSA | ✅ | ✅ | On |
| Q-learning | ✅ | ✅ | Off |
| Double Q-learning | ✅ | ✅ | Off |

---

## 📚 Reference

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
