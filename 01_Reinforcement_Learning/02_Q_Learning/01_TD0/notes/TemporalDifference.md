# Temporal Difference

## Overview

TD methods combine the **model-free sampling** of MC with the **online bootstrapping** of DP. They update value estimates at **every step** without waiting for episode completion.

---

## Key Concept: TD Error

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

The TD error is the difference between the estimated value and the bootstrapped target. This is the core signal used for all TD updates.

---

## Comparison: DP vs MC vs TD

| | DP | MC | TD |
|--|----|----|-----|
| Model required | ✅ | ❌ | ❌ |
| Bootstrapping | ✅ | ❌ | ✅ |
| Online updates | ✅ | ❌ | ✅ |
| Wait for episode end | ❌ | ✅ | ❌ |

---

## Algorithms

### 1. TD(0) Prediction

Estimate $V^\pi$ under a fixed policy, updating at every step.

**Update rule:**
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

```python
td = TD0(env)
V = td.evaluation(policy, alpha=0.1, gamma=0.99, max_iter=10000)
```

---

### 2. SARSA (On-policy TD Control)

Learn $Q^\pi$ using the **same ε-greedy policy** for both action selection and update.

**Update rule:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

Named after the tuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.

Converges to **ε-soft optimal policy**, not true $\pi^*$.

```python
policy, Q, history = td.SARSA(alpha=0.1, gamma=0.99, max_iter=10000)
```

---

### 3. Expected SARSA

Use the **expected value** over all actions instead of sampling the next action.

**Update rule:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a} \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

Lower variance than SARSA because it removes the randomness of action sampling.

```python
policy, Q, history = td.ESARSA(alpha=0.1, gamma=0.99, max_iter=10000)
```

---

### 4. Q-learning (Off-policy TD Control)

Learn $Q^*$ directly using a **greedy target** regardless of the behavior policy.

**Update rule:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

- **Behavior policy:** ε-greedy (exploration)
- **Target:** greedy max (learning)

Converges to $Q^*$ regardless of which policy is followed.

```python
policy, Q, history = td.QL(alpha=0.1, gamma=0.99, max_iter=10000)
```

---

### 5. Double Q-learning

Q-learning overestimates values due to **maximization bias** — using max for both action selection and evaluation.

Double Q-learning maintains two Q-tables and decouples selection from evaluation:

$$Q_1(S, A) \leftarrow Q_1(S, A) + \alpha \left[ R + \gamma Q_2(S', \arg\max_{a'} Q_1(S', a')) - Q_1(S, A) \right]$$

Each update randomly chooses which Q to update (50/50).

```python
policy, Q, history = td.DQL(alpha=0.1, gamma=0.99, max_iter=10000)
```

---

## Algorithm Comparison

| Algorithm | Policy | Target | Converges to |
|-----------|--------|--------|--------------|
| TD(0) | Given $\pi$ | $V(s')$ | $V^\pi$ |
| SARSA | ε-greedy | $Q(s', a')$ from same policy | $Q^{\pi_\varepsilon}$ |
| Expected SARSA | ε-greedy | $\mathbb{E}_\pi[Q(s',a')]$ | $Q^{\pi_\varepsilon}$ |
| Q-learning | ε-greedy (behavior) | $\max_{a'} Q(s',a')$ | $Q^*$ |
| Double Q-learning | ε-greedy (behavior) | $Q_2(s', \arg\max Q_1)$ | $Q^*$ (less biased) |

---

## ε Decay Schedule

Using $\varepsilon = 1/t$ ensures:
- Early episodes: high exploration
- Later episodes: more greedy → approaches $\pi^*$

```python
epsilon = 1 / n_iter
```
