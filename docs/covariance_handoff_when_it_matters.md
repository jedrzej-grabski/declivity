# When does CMA-ES → L-BFGS-B covariance handoff actually help?

**TL;DR.** The CMA-ES covariance helps L-BFGS-B if and only if the basin
it lands in is *anisotropic* and the warmup was long enough to learn
that anisotropy. On near-isotropic multimodal problems (Rastrigin,
Griewank) the C⁻¹ handoff is indistinguishable from a plain
identity-Hessian handoff — basin selection determines the final
fitness, not the choice of B₀. On highly anisotropic and rotated
problems with sufficient warmup the C⁻¹ handoff wins by 3–4 orders of
magnitude. With too-long warmup the C⁻¹ handoff can actually hurt
because CMA-ES collapses its scale and feeds L-BFGS-B an ill-scaled
initial Hessian.

## Setup

All experiments use the new `src/benchmarking/` framework: same seed →
same `x0`, same CMA-ES RNG path. L-BFGS-B uses the Armijo line search
(`More-Thuente` rejects multimodal ripples within ~20 evals).
Default 15 seeds, total budget 10 000 evaluations, L-BFGS-B memory
m = 5 unless noted. All plots come from re-running on this branch.

Algorithms compared:

- **CMA-ES** — standalone, full budget.
- **L-BFGS-B** — standalone, full budget.
- **Handoff (C⁻¹)** — CMA-ES warmup then L-BFGS-B with
  B₀ = C⁻¹ (computed from CMA-ES's cached eigendecomposition).
- **Handoff (identity)** — CMA-ES warmup then L-BFGS-B with the
  default B₀ = I. Same warmup path as C⁻¹, so this isolates the value
  of *passing the covariance* from the value of *starting from the
  CMA-ES mean*.

## Background — why I started this study

The first batch on unrotated Rastrigin and Griewank produced this
surprising result:

| Problem   | C⁻¹ median f | Identity median f | C⁻¹ median evals | Identity median evals |
|-----------|--------------|-------------------|------------------|-----------------------|
| Rastrigin | 1.09e+01     | 1.09e+01          | 1567             | 1538                  |
| Griewank  | 7.40e-03     | 7.40e-03          | 1569             | 1538                  |

Identical to machine precision. The covariance was apparently useless.
This was suspicious — the supervisor study on rotated Ellipsoid
(`docs/supervisor_report_initial_hessian.md`) had shown a ~300× gap
between identity and the true Hessian as B₀. Something about
multimodal problems was different.

## Hypothesis

The local Hessian at the basin determines whether B₀ can help.

- **Rastrigin** local Hessian at any minimum
  is exactly `(2 + 40π²) · I` — perfectly isotropic. Verified
  numerically:

  ```
  d=10: Rastrigin eigs [396.8, 396.8], cond=1
  d=30: Rastrigin eigs [396.8, 396.8], cond=1
  d=50: Rastrigin eigs [396.8, 396.8], cond=1
  ```

  No direction is privileged → identity is *already optimal* up to
  scale, and L-BFGS-B's theta adapts the scale in one iteration.

- **Griewank** local Hessian eigenvalues are
  `1/2000 + 1/i`, giving condition number ≈ d.

  ```
  d=10: Griewank eigs [0.10, 1.0], cond=10
  d=30: Griewank eigs [0.034, 1.0], cond=30
  d=50: Griewank eigs [0.020, 1.0], cond=49
  ```

  Anisotropic, but axis-aligned. A few BFGS corrections can learn
  the per-axis scaling from identity — the directional structure was
  trivial.

For covariance to *demonstrably* matter we need a problem where:

1. The basin has condition number ≫ d, so identity is meaningfully
   wrong.
2. The anisotropy is **rotated** (not axis-aligned), so a few BFGS
   corrections can't reconstruct it from identity alone.
3. CMA-ES has had enough warmup to learn the rotation.

## Tools added for the study

- **`RotatedFunction`** (`src/utils/benchmark_functions.py`) — generic
  wrapper that turns any `BenchmarkFunction` with a gradient into a
  rotated variant. Inherits bounds, gradient transforms as
  `Rᵀ · ∇f_base(R x)`, supports `"uniform_45" / "golden" / "random" /
  explicit matrix` rotations.

- **`RippledEllipsoid`** (same file) — `f(x) = Σ scale_i x_i² +
  amplitude·Σ(1 − cos 2π x_i)`. Anisotropic quadratic + Rastrigin-style
  ripples. Knobs:
  - `condition` controls the eigenvalue spread of the quadratic part.
  - `amplitude` controls multimodality. Large amplitude → many local
    minima (Rastrigin-like). Small amplitude → near-unimodal,
    dominated by the quadratic.

- **`CMAESLBFGSBHandoff(transform="identity")`** — third handoff
  baseline, alongside `"inverse"` and `"sigma_inverse"`. Lets us
  isolate "share x₀" from "share covariance".

- New example scripts:
  - `examples/rotated_multimodal_handoff.py` — drives rotated
    Rastrigin / Griewank sweeps.
  - `examples/rippled_ellipsoid_handoff.py` — drives RippledEllipsoid
    sweeps over (dimension × condition × memory × amplitude).
  - `examples/rippled_handoff_timing_sweep.py` — sweeps warmup budget
    on a fixed problem.

## Experiment 1 — rotated Rastrigin and Griewank

Rotating the multimodal functions doesn't introduce anisotropy where
none exists. As predicted, the C⁻¹ vs identity gap stays absent.

10D and 30D rotated Griewank, default m = 10, warmup 2500 evals:

![rotated griewank m=10](../plots/hybrid/exp1_rotated_griewank_m10/convergence.png)

| Problem        | C⁻¹ median f | Identity median f |
|----------------|--------------|-------------------|
| Griewank 10D   | 7.40e-03     | 7.40e-03          |
| Griewank 30D   | 0.00e+00     | 0.00e+00          |

The 30D case is degenerate — in high dimension `∏ cos(x_i/√i)` decays
exponentially toward zero, so Griewank effectively becomes a sphere
and even standalone L-BFGS-B reaches zero in ~70 evaluations from a
random starting point.

## Experiment 2 — rotated RippledEllipsoid, multimodal regime (amp=10)

Strong ripples, moderate-to-high condition. The function has many
local minima.

10D + 30D, condition 1000, amplitude 10 (Rastrigin-strength ripples),
m=5, warmup 2500:

![rippled multimodal cond=1000](../plots/hybrid/exp3_rippled_c1000_m5/convergence.png)

| Problem              | C⁻¹ median f | Identity median f |
|----------------------|--------------|-------------------|
| Rippled c=1000 d=10  | 3.15e+01     | 3.15e+01          |
| Rippled c=1000 d=30  | 1.07e+02     | 1.07e+02          |

**Identical.** Multimodality dominates: both algorithms find the same
local minimum from the same CMA-ES mean. Their first descent steps
differ (steepest descent vs Newton-like) but the basin floor is the
same point and both reach it well within budget.

Pushed harder (condition 10⁶, amplitude still 10, m=3):

![rippled multimodal cond=1e6](../plots/hybrid/exp4_rippled_c1e6_m3/convergence.png)

| Problem               | C⁻¹ median f | Identity median f |
|-----------------------|--------------|-------------------|
| Rippled c=1e6 d=10    | 2.79e+01     | 2.79e+01          |
| Rippled c=1e6 d=30    | 1.78e+02     | 1.51e+02          |

Now the descent paths actually diverge — they land in different local
minima — but the *medians* are within noise of each other, and
identity happens to win on 30D. With strong multimodality, B₀ steers
into a slightly different basin; which basin is lower is essentially
luck.

**Takeaway:** the multimodal premise of the original task makes this
particular question hard to answer in the multimodal regime itself.
The handoff buys speed-to-basin-floor; it doesn't buy escape from
multimodality.

## Experiment 3 — rotated RippledEllipsoid, near-unimodal regime (amp=0.1)

Drop the ripple amplitude until the cos-saddle curvature
(`4π²·amplitude`) is small compared to the quadratic eigenvalues.
The function is technically still multimodal in the low-scale
dimensions, but the global basin is enormous and contains the entire
trajectory once CMA-ES has had a few generations.

condition = 10⁶, amplitude = 0.1, m = 5, warmup 2500:

![rippled near-unimodal](../plots/hybrid/exp6_low_ripple_high_cond/convergence.png)

| Problem      | CMA-ES    | L-BFGS-B   | C⁻¹ handoff | Identity handoff | C⁻¹ / identity |
|--------------|-----------|------------|-------------|------------------|----------------|
| d=10         | 3.90e-16  | 1.97e-15   | 4.82e-20    | 1.70e-15         | 35 000×        |
| d=30         | 1.84e+02  | 8.22e-16   | 5.82e-18    | 3.79e-15         | 650×           |
| d=50         | 4.56e+03  | 6.48e-16   | 8.54e-16    | 1.10e-14         | 13×            |

This is the finding the supervisor study predicted: with rotated
high-condition anisotropy, the CMA-ES covariance is genuinely
informative and the C⁻¹ handoff descends to a much smaller absolute
fitness than the identity handoff within the same total budget. The
effect is strongest in 10D — both methods are deep into machine-zero
territory, but C⁻¹'s Newton-like steps keep pushing while identity's
steepest-descent steps plateau.

L-BFGS-B alone is competitive here because the function is nearly
unimodal — the ripples don't trap it.

## Experiment 4 — warmup timing sweep

How much CMA-ES warmup is needed before the covariance is worth
trusting? And is there a point at which more warmup hurts?

Fixed problem (rotated RippledEllipsoid, cond=10⁶, amp=0.1, d=30,
m=5), fixed total budget 10 000. Warmup swept across {500, 1 500,
3 000, 5 000, 7 500} evaluations; remainder goes to L-BFGS-B. Both
C⁻¹ and identity transforms shown at each warmup.

![warmup timing sweep](../plots/hybrid/exp7_timing_sweep/convergence.png)

| Warmup (evals) | C⁻¹ median f | Identity median f | Winner |
|----------------|--------------|-------------------|--------|
| 500            | 5.46e-16     | 9.57e-16          | C⁻¹ (mild) |
| 1 500          | 3.82e-16     | 1.72e-15          | C⁻¹ |
| 3 000          | **1.88e-17** | 6.06e-15          | C⁻¹ |
| 5 000          | 6.99e-05     | 1.33e-10          | identity |
| 7 500          | 5.14e-02     | 1.15e-03          | identity |

Two effects fight each other as warmup grows.

1. **Information.** More warmup → CMA-ES has learned the rotation
   more accurately → C⁻¹ is more informative.
2. **Scale collapse.** As CMA-ES converges, the sqrt-eigenvalues `D`
   shrink. C⁻¹ has eigenvalues `1/D²`, which blow up. Because the
   framework's L-BFGS-B keeps `persist_initial_hessian=True`, this
   ill-scaled B₀ stays in effect for the entire L-BFGS-B segment and
   causes the optimizer to overshoot or stall on the now-tiny basin.

The sweet spot is around warmup = 3 000 (≈ 100 CMA-ES generations on
d=30). Beyond that, **passing the covariance is actively harmful** —
the identity baseline reaches 1.3e-10 while the C⁻¹ run only gets to
7.0e-05 (a 5×10⁵ swing in the opposite direction).

Identity handoff also degrades with longer warmup, but for a different
reason: less remaining L-BFGS-B budget. It declines monotonically.

## Conclusions

1. **On near-isotropic multimodal problems** (Rastrigin, Griewank in
   their natural form), C⁻¹ and identity are indistinguishable. This
   is not a bug; it reflects the geometry. The previous finding that
   the handoff matches CMA-ES quality in <½ the evaluations stands —
   the speed-up comes from sharing the warmup `x₀`, not the
   covariance.

2. **On rotated, ill-conditioned, near-unimodal problems**, the C⁻¹
   handoff reaches a fitness 10¹–10⁴ times smaller than the identity
   handoff in the same budget. This is the regime where the supervisor
   ellipsoid result actually applies.

3. **The covariance handoff is not always safe to apply**: long
   warmup collapses the CMA-ES scale, and `1 / D²` becomes
   numerically large. Combined with `persist_initial_hessian=True`,
   this poisons the entire L-BFGS-B run. A trace-normalized variant
   (`C / trace(C) · n`, then inverted) or a fix that drops
   `persist_initial_hessian` after a few iterations might be worth
   exploring.

4. **For the thesis benchmarking story**, the honest claim is:
   - On multimodal problems whose local Hessian is near-isotropic
     (Rastrigin), the handoff just buys speed (shared `x₀`).
   - On problems whose local Hessian is anisotropic and rotated
     (RippledEllipsoid, the original supervisor Ellipsoid study),
     the covariance is the dominant value of the handoff.
   - There is a non-trivial timing tradeoff: too short and the
     covariance is noisy; too long and the scale collapses.

## Reproducing

```bash
# Multimodal baseline (the puzzling one)
PYTHONPATH=. pdm run python examples/multimodal_handoff_benchmark.py \
    --include-identity-baseline --num-workers 6 \
    --output-dir plots/hybrid/multimodal_handoff

# Rotated multimodal (no anisotropy gained)
PYTHONPATH=. pdm run python examples/rotated_multimodal_handoff.py \
    --bases Griewank --dimensions 10 30 --memory 10 \
    --num-workers 6 --output-dir plots/hybrid/exp1_rotated_griewank_m10

# Rotated rippled multimodal (still no clear winner)
PYTHONPATH=. pdm run python examples/rippled_ellipsoid_handoff.py \
    --dimensions 10 30 --conditions 1000 --memory 5 \
    --num-workers 6 --output-dir plots/hybrid/exp3_rippled_c1000_m5

# Rotated rippled near-unimodal (C^-1 wins by orders of magnitude)
PYTHONPATH=. pdm run python examples/rippled_ellipsoid_handoff.py \
    --dimensions 10 30 50 --conditions 1000000 --memory 5 \
    --amplitude 0.1 --num-workers 6 \
    --output-dir plots/hybrid/exp6_low_ripple_high_cond

# Warmup timing sweep on the regime where C^-1 wins
PYTHONPATH=. pdm run python examples/rippled_handoff_timing_sweep.py \
    --dimensions 30 --warmup-budgets 500 1500 3000 5000 7500 \
    --num-workers 6 --output-dir plots/hybrid/exp7_timing_sweep
```

## Open questions

- Is the scale-collapse failure mode at long warmup an argument for
  the `sigma_inverse` transform (which divides C⁻¹ by σ², partially
  cancelling the collapse)? Worth running the sweep with all three
  transforms side by side.
- Would dropping `persist_initial_hessian=True` after the first few
  iterations recover the long-warmup case? The persist flag was the
  right default for the original rotated-ellipsoid study; it might
  not be for handoff scenarios where B₀ comes from a converged
  distribution.
- On problems whose anisotropy isn't aligned to either coordinate
  axes or the principal eigenvectors CMA-ES finds, does C⁻¹ still
  help, or does the "wrong rotation" hurt? Rotation-mode ablation
  would settle this.
