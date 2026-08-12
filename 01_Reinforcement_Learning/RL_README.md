# 01_Reinforcement_Learning

Reinforcement Learning study from fundamentals to advanced topics, following Sutton & Barto and recent Deep RL literature, organized by the OpenAI Spinning Up taxonomy (Model-Free RL: Policy Optimization / Q-Learning).

---

## 📁 Structure

```
01_Reinforcement_Learning/
├── 00_Fundamentals/  # DP, MC (Dynamic Programming, Monte Carlo)
├── 01_Policy_Optimization/
│   ├── 01_Policy_Gradient/
│   │   └── 01_REINFORCE/  # REINFORCE, REINFORCE+Baseline
│   ├── 02_A2C_A3C/
│   │   └── 01_Vanilla_AC/  # Vanilla Actor-Critic (1-step TD)
│   └── 03_TRPO/  # Trust Region Policy Optimization
└── 02_Q_Learning/
    └── 01_TD0/  # TD(0) prediction, SARSA, Q-learning, Expected SARSA, Double Q-learning
```
## 📌 Topics

| Folder | Topic |
|--------|-------|
| `00_Fundamentals` | GridWorld, DP, MC, TD(0) prediction |
| `01_Policy_Optimization/01_Policy_Gradient/01_REINFORCE` | REINFORCE, REINFORCE+Baseline (variance comparison) |
| `01_Policy_Optimization/02_A2C_A3C/01_Vanilla_AC` | Vanilla Actor-Critic, TD bootstrapping instability |
| `01_Policy_Optimization/03_TRPO` | Natural gradient, conjugate gradient, line search |
| `02_Q_Learning/01_TD0` | SARSA, Q-learning, Expected SARSA, Double Q-learning |

---

## 🔍 Key Findings

- **REINFORCE vs Baseline**: Baseline reduces gradient variance and speeds up
  early learning, but does not prevent eventual policy collapse — collapse
  timing appears largely stochastic rather than baseline-dependent
  (see `01_Policy_Gradient/01_REINFORCE/notes/`).
- **Vanilla Actor-Critic**: TD bootstrapping with a shared function
  approximator can cause the value function to overshoot actual achievable
  performance, oscillate, and drive the policy into an unrecoverable
  saturated state — motivating trust-region constraints (see `02_A2C_A3C/01_Vanilla_AC/notes/`).
- **TRPO**: KL-constrained natural gradient updates avoided the collapse
  observed in Actor-Critic, solving CartPole-v1 in 10 iterations
  (~20,480 steps) with every line search succeeding on the first try
  and KL divergence always within the trust region (see `03_TRPO/notes/`).

---

## 📖 Reference

**00_Fundamentals**
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

**01_Policy_Optimization/01_Policy_Gradient/01_REINFORCE**
- Williams, R. J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. Machine Learning, 8(3-4), 229-256.

**01_Policy_Optimization/02_A2C_A3C/01_Vanilla_AC**
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Ch. 13 (Policy Gradient Methods / Actor-Critic).

**01_Policy_Optimization/03_TRPO**
- Schulman, J., et al. (2015). *Trust Region Policy Optimization*.
- Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*.