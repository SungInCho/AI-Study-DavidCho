# Dynamic Programming

## Overview

DP methods solve MDPs by using **full knowledge of environment dynamics** $P(s'|s,a)$ and $R(s,a,s')$. They are not model-free — they require a complete model of the environment.

---

## Key Concept: Bellman Equations

### Bellman Expectation Equation
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

### Bellman Optimality Equation
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

---

## Algorithms

### 1. Policy Evaluation

Iteratively compute $V^\pi$ for a given policy $\pi$.

**Update rule:**
$$V(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R + \gamma V(s') \right]$$

**Convergence:** repeat until $\max_s |V_{new}(s) - V(s)| < \theta$

```python
dp = DP_policyeval(env)
V = dp.eval(policy, gamma=0.99, theta=1e-6)
```

---

### 2. Policy Iteration

Alternate between Policy Evaluation and Policy Improvement until policy converges.

**Policy Improvement (Greedy):**
$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R + \gamma V(s') \right] = \arg\max_a Q(s,a)$$

**Algorithm:**
```
Initialize π randomly
Repeat:
    V ← PolicyEvaluation(π)
    π' ← GreedyPolicy(V)
    if π' == π: stop
    π ← π'
```

```python
dp = DP_policy_iteration(env)
policy, V = dp.forward(gamma=0.99)
```

---

### 3. Value Iteration

Combine one step of evaluation and improvement in a single update.

**Update rule:**
$$V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a) \left[ R + \gamma V(s') \right]$$

More efficient than Policy Iteration — no need to wait for full evaluation to converge.

```python
vi = DP_valueiter(env)
policy, V = vi.forward(gamma=0.99)
```

---

## Comparison

| Algorithm | Evaluation Steps | Convergence |
|-----------|-----------------|-------------|
| Policy Evaluation | Until convergence | $V^\pi$ |
| Policy Iteration | Until convergence per step | $V^*$, $\pi^*$ |
| Value Iteration | 1 step | $V^*$, $\pi^*$ |

Policy Iteration and Value Iteration both converge to the same $V^*$ and $\pi^*$.

---

## Limitations

- Requires full knowledge of $P(s'|s,a)$ and $R$
- Not applicable to real-world environments where dynamics are unknown
- Computationally expensive for large state spaces (curse of dimensionality)
