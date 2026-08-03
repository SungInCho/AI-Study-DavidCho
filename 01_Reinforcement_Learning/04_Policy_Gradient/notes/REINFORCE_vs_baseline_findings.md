## Findings: High Variance in REINFORCE

Trained REINFORCE on CartPole-v1 (solved at episode 208, 100-episode 
avg reward ≥ 195). However, further analysis reveals significant 
variance in the learned policy's performance.

### Evidence

**1. Training curve (last 100 episodes before "solved")**
Individual episode rewards ranged from 19 to 500, despite the 
100-episode average exceeding the solved threshold. Mean ≈ 195, 
but std is large and min/max span nearly the entire possible range.

**2. Evaluation over 20 fresh episodes (same trained policy)**
| Mode     | Mean  | Std  | Min | Max |
|----------|-------|------|-----|-----|
| Greedy   | 272.1 | 24.2 | 237 | 316 |
| Sampling | 226.3 | 47.7 | 139 | 331 |

Even with a fixed, fully trained policy, episode outcomes vary 
substantially — greedy action selection reduces but does not 
eliminate this variance.

### Why this happens

1. **Monte Carlo return estimation**: G_t is computed from a single 
   full trajectory, so it inherits all the stochasticity of the 
   environment and the sampled actions.
2. **Poor credit assignment**: every action within an episode is 
   scaled by the same G_t, so lucky/unlucky moments get conflated 
   with actual policy quality.
3. **No baseline**: since CartPole rewards are always positive, 
   raw G_t only pushes action probabilities up, with no notion of 
   "better or worse than expected."
4. **One gradient step per episode**: no averaging across multiple 
   trajectories before each update.

---

## Part 2: Adding a Baseline (Advantage) — Reduces Variance, Speeds Learning

### Setup
Added a separate `ValueNetwork` V(s) trained via MSE loss to predict
G_t. The policy gradient was updated to use the advantage
A_t = G_t − V(s_t) instead of raw G_t, with V(s_t) detached before the
policy loss (so no gradient flows from the policy loss into the value
network).

### Result (early-stopped comparison)
REINFORCE with baseline solved at episode 206 — comparable convergence
speed to vanilla (208). This alone does not show a clear variance
reduction, because the two runs stop at different training stages
(comparing apples to oranges).

### Result (fixed 500-episode comparison — fairer test)
Training both algorithms for a fixed 500 episodes (no early stopping)
revealed a much clearer picture: REINFORCE with baseline learns
**faster** in the early-to-mid stage (reaching ~450 average reward by
episode ~350, vs. vanilla only reaching similar levels around episode
~400-450). This confirms that reducing gradient variance via a
baseline does accelerate learning, consistent with theory.

---

## Part 3: Both Algorithms are susceptible to Gradient Collapse 

### Initial observation (500 episodes, fixed)
In an earlier 500-episode run, REINFORCE with baseline collapsed after
reaching high performance (peaking near 450-460 around episode 350-400,
then dropping to ~220 by episode 500), while vanilla REINFORCE showed
no collapse in the same window — it was still on an upward trajectory,
reaching ~450 by episode 500. This initially suggested that the
baseline might make policy collapse *worse* or *earlier*, since
reducing gradient variance lets the policy converge to a confident
(and therefore fragile) state faster.

### Extended observation (1000 episodes, fixed, seeded for reproducibility)
Extending training to 1000 episodes overturned this initial
interpretation. With more training time, **both algorithms eventually
collapsed**:

- **Vanilla REINFORCE**: reached a first local peak (~320) around
  episode 300, dipped to ~150 between episodes 400-600, recovered to a
  second peak (~490) around episode 750-950, then collapsed to ~330 by
  episode 1000.
- **REINFORCE with Baseline**: climbed steadily and maintained a long
  stable plateau (~450-490) from roughly episode 500 to 900 — notably
  *more* stable than vanilla over that stretch — before also
  collapsing, dropping to ~250 by episode 1000.

Final evaluation (20 episodes, greedy) at episode 1000, after both
policies had already collapsed:

| Algorithm            | Mean  | Std  | Min | Max |
|-----------------------|-------|------|-----|-----|
| Vanilla (1000 eps)     | 99.3  | 45.7 | 18  | 131 |
| Baseline (1000 eps)    | 165.7 | 6.7  | 154 | 179 |

This changes the conclusion from the initial 500-episode observation:
**the baseline does not collapse "first" or "worse" as a rule.** In
the 500-episode run baseline happened to collapse within the observed
window and vanilla did not yet; in the 1000-episode run, baseline in
fact sustained high performance *longer* than vanilla (a ~400-episode
stable plateau vs. vanilla's more erratic path with an intermediate
dip). The timing of collapse appears to be a largely stochastic
event — determined by when a noisy gradient happens to hit an
already-confident policy — rather than a deterministic consequence of
using a baseline or not.

### What does remain consistent: baseline still reduces variance
Even though baseline does not prevent eventual collapse, its
variance-reduction role remains clearly visible in every measurement:

- **During stable training**, baseline's moving-average curve is
  visibly smoother and sustains near-peak performance over a much
  longer stretch of episodes than vanilla.
- **Gradient norm** (tracked via total policy gradient norm before each
  optimizer step) shows the same-order-of-magnitude spikes for both
  algorithms throughout training — meaning the *frequency and size* of
  noisy gradients is not what baseline changes.
- **Even after collapse**, the two algorithms differ sharply in
  variance: vanilla's post-collapse evaluation std (45.7) remains large
  and erratic, while baseline's post-collapse std (6.7) is small — the
  policy converged to a *consistently* poor behavior rather than an
  erratic one. This is the same variance-reduction effect as before,
  just applied to a degraded policy instead of a good one.

### Why this happens

**Factor A — Monte Carlo gradient noise (trigger, present throughout
training)**
Even for a well-trained, high-performing policy, an unlucky episode's
G_t can still produce a large-magnitude, noisy gradient. The gradient
norm plot shows this noise is present continuously across training for
both algorithms — it does not appear only near collapse points.

**Factor B — No constraint on policy update size (amplifier)**
The update rule θ ← θ + α∇J(θ) has no mechanism limiting how much the
*action distribution* itself can change per step. Once a policy becomes
confident (e.g. probabilities near [0.99, 0.01]), a small parameter
change can correspond to a large, destabilizing shift in behavior.

**Why collapse timing looks stochastic rather than baseline-dependent:**
Because Factor A (noisy gradients) occurs continuously and
unpredictably for both algorithms, whether a given noisy gradient
triggers collapse depends on whether it happens to arrive while the
policy is in the fragile, overconfident regime described in Factor B.
Baseline reduces the *magnitude* of variance on average, but does not
eliminate the occasional large spike, and does not track or limit how
confident (and thus fragile) the policy has become. As a result,
collapse can occur "early" or "late" for either algorithm depending on
when a large-enough gradient spike coincides with a highly confident
policy state — an essentially stochastic coincidence rather than a
deterministic property of having (or not having) a baseline.

---

## Conclusion and Next Step

Baseline (advantage-based REINFORCE) successfully and consistently
reduces gradient variance — visible in faster early learning, longer
stable high-performance plateaus, and even in how consistently (if
poorly) the policy behaves after collapse. However, across both the
500- and 1000-episode experiments, baseline does **not** prevent
eventual policy collapse, and the timing of that collapse does not
follow a simple "baseline is safer" or "baseline collapses first"
rule — it appears to be a largely stochastic event driven by whether a
noisy gradient happens to strike an already-overconfident policy.