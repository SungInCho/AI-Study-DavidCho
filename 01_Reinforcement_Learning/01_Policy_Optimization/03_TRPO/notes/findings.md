# TRPO Findings: CartPole-v1

## Summary
Implemented TRPO (Trust Region Policy Optimization) from scratch —
natural gradient via conjugate gradient, KL-constrained step size, and
backtracking line search — and compared its behavior against REINFORCE,
REINFORCE+Baseline, and Vanilla Actor-Critic on CartPole-v1. TRPO solved
the environment in 10 iterations (~20,480 environment steps), far faster
and with no observed instability, in contrast to Actor-Critic's collapse
after ~250-400 episodes in earlier experiments.

## Results

### Convergence speed
| Algorithm            | Steps/Episodes to solve |
|-----------------------|--------------------------|
| Vanilla REINFORCE      | 208 episodes |
| REINFORCE + Baseline   | 206 episodes |
| Vanilla Actor-Critic   | 255 episodes (later collapsed under extended training) |
| TRPO                   | 10 iterations (~20,480 steps) |

### KL divergence stayed within the trust region at every update
Across all 10 iterations, the realized KL divergence between old and
new policy ranged from 0.0038 to 0.0082 — always below max_kl=0.01.
Every line search succeeded on the first try (0 backtracks every
iteration), meaning the natural-gradient step size (computed via
sqrt(2*max_kl / (x^T F x))) was accurate enough on its own to satisfy
the constraint without needing to shrink the step.

### Value loss increased over training — and this is expected, not a failure
Value loss rose from ~95 to ~255 over the 10 iterations. Unlike the
value divergence observed in Actor-Critic (where V grew faster than
actual achievable reward and then crashed), here the increase reflects
returns themselves growing as the policy improves and episodes get
longer — the value network is tracking a genuinely growing target, not
diverging from it.

### Surrogate loss stayed near zero by construction
The surrogate objective (importance-sampling ratio × advantage)
measured at the start of each iteration was consistently near zero
(~1e-7 scale). This is expected: each iteration begins with fresh data
collected under the *current* policy, so ratio ≈ 1 at that point, and
advantages are normalized to zero mean — making the loss itself an
uninformative absolute quantity. The gradient of this loss (used to
compute the update direction) is still meaningful; the loss value alone
is not a useful convergence signal for TRPO.

## Why TRPO avoided the collapse seen in Actor-Critic

Actor-Critic's failure mode (documented in `vanilla_actor_critic`
findings) was traced to two interacting factors:
- Factor A: noisy gradients from TD bootstrapping
- Factor B: no constraint on how much a single update could change the
  policy's action distribution

TRPO directly targets Factor B. Every update is explicitly scaled so
that the KL divergence between the old and new policy stays within
max_kl, and line search verifies this empirically (not just via the
2nd-order approximation) before accepting a step. In this experiment,
that constraint was never violated, and no single update was allowed to
push the policy toward the kind of saturated, fragile state that
preceded collapse in Actor-Critic.

## Caveats
- Single seed, single run — collapse in Actor-Critic experiments was
  shown to have stochastic timing across seeds/hyperparameters, so a
  single successful TRPO run does not rule out failure under different
  conditions. Repeating with multiple seeds and longer training
  (well past "solved") would be needed to confirm TRPO's stability
  claim as rigorously as the Actor-Critic collapse was confirmed.
- CartPole is a simple, short-horizon environment where REINFORCE and
  Baseline also eventually reached high performance (>400) given
  enough episodes — TRPO's main demonstrated advantage here is sample
  efficiency and stability, not reaching a fundamentally higher final
  performance ceiling.

## Next step
PPO — approximates the same trust-region idea via a simpler clipped
surrogate objective, avoiding the conjugate gradient / Fisher-vector
product machinery while retaining (approximately) the same update-size
safety guarantee.