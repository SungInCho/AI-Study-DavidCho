# GridWorld

## Overview

A custom grid-based environment that satisfies the **Markov Property**, used as the foundation for all RL algorithm experiments in this repository.

---

## Markov Property

A state $s$ satisfies the Markov Property if:

$$P(S_{t+1} | S_t) = P(S_{t+1} | S_1, S_2, ..., S_t)$$

The future depends only on the **current state**, not the history. GridWorld satisfies this because the next state is determined solely by the current position and action.

---

## Environment Design

### State Space
- Grid of size $n \times n$
- State: $(r, c)$ where $r$ is row, $c$ is column
- Flattened index: $r \times n + c$

### Action Space
- 4 actions: Up (0), Down (1), Left (2), Right (3)

### Reward Structure
| Event | Reward |
|-------|--------|
| Reach goal | +1 |
| Hit wall / obstacle | -1 |
| Normal step | -0.1 |

### Special Cells
- **Obstacles**: Agent cannot enter. Attempting to move into one results in staying in place with -1 reward.
- **Icy Floors**: Stochastic transitions. When on an icy floor:
  - 80% probability: move in intended direction
  - 10% probability: slip perpendicular (left)
  - 10% probability: slip perpendicular (right)

---

## Transition Probability

For deterministic cells:

$$P(s'|s, a) = 1 \text{ for the intended next state}$$

For icy floor cells:

$$P(s'|s, a) = \begin{cases} 0.8 & s' = \text{intended} \\ 0.1 & s' = \text{perpendicular}_1 \\ 0.1 & s' = \text{perpendicular}_2 \end{cases}$$

If the intended next state is a wall or obstacle, the agent stays in place and that probability is added to $P(s|s,a)$.

---

## Key Methods

```python
env = GridWorld(size=5, obstacles=[(1,1)], icy_floors=[(2,2)])

env.reset()                          # → start state (0,0)
env.step(action)                     # → (next_state, reward, move, done)
env.get_transition_prob(state, action) # → {next_state: (prob, reward, move)}
env.get_dynamics_rewards()           # → dynamics (S,A,S'), rewards (S,A,S')
env.render()                         # → ASCII grid visualization
env.state_to_index(state)            # → int index
env.index_to_state(index)            # → (r, c) tuple
```

---

## MDP Components

| Symbol | Meaning | In GridWorld |
|--------|---------|--------------|
| $\mathcal{S}$ | State space | All $(r,c)$ positions |
| $\mathcal{A}$ | Action space | {Up, Down, Left, Right} |
| $P(s'\|s,a)$ | Transition probability | Deterministic or icy |
| $R(s,a,s')$ | Reward function | -0.1, -1, +1 |
| $\gamma$ | Discount factor | 0.99 |
