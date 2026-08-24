# Hessian-completed Nelder-Mead

Status: **proof of concept, promoted to a registered optimizer.** The mechanism
works and is measured; whether it is *useful in the CMA-ES handoff* is still
open, and the current evidence says probably not yet. Read the
[Where this stands](#where-this-stands) section before building on it.

---

## 1. The problem this started from

Seeding Nelder-Mead's initial simplex from a CMA-ES covariance — the mechanism
`CovarianceSimplexInitializer` implements and the `neldermead` arm of exp1
measures — underdelivers, and no amount of tuning `base_size` / `ratio_floor`
fixes it. The reason is structural rather than numerical.

Every classic Nelder-Mead move — reflect, expand, contract, shrink — is a
**fixed affine combination of the current vertices**, with coefficients that
depend on neither the objective nor any Hessian:

```
x_r = (1 + rho) * xbar - rho * v_worst        rho, chi, psi, sigma are constants
```

So Nelder-Mead is exactly affine-invariant: run it on `f` from simplex `S`, and
on `f(A y)` from `A^-1 S`, and the two trajectories are exact affine images of
one another. The corollary is sharp:

> **Classic Nelder-Mead is structurally incapable of consuming curvature.** The
> only place a Hessian can enter is the initial simplex, and that is merely a
> change of variables — consumed once, then forgotten. Any modification that
> mixes the vertices with coefficients not derived from `H` is invisible to `H`.

That is not just a theoretical objection. Measured on the `10^6`-conditioned
Ellipsoid at `d = 10`, handing the *exact* Hessian to the simplex-shaping
mechanism makes it **10^4x worse** than the isotropic default, with runs
stalling early on a collapsed simplex. Shaping helps for mid-quality
covariances and hurts for sharp ones: it degrades precisely when the covariance
is most anisotropic, because that is when the simplex it builds is closest to
degenerate — and Nelder-Mead has no move that restores geometry.

## 2. The idea: Hessian completion

A quadratic model over `n` dimensions has `1 + n + n(n+1)/2` free parameters.
The simplex's `n + 1` known values cannot come close to pinning that down, which
is why nobody fits a quadratic to a simplex.

**But if the Hessian is *given*, only the constant and the gradient are left:
`1 + n` unknowns against `n + 1` equations. The system is square.**

With `v_0` the best vertex and `d_i = v_i - v_0`:

```
m(x)   = f_0 + g'(x - v_0) + 1/2 (x - v_0)' H (x - v_0)
m(v_i) = f_i    <=>    g' d_i = f_i - f_0 - 1/2 d_i' H d_i  =:  b_i
D g = b,        D = [d_1; ...; d_n]   (n x n)
```

`g` is a **curvature-corrected simplex gradient**: the plain simplex gradient
Nelder-Mead implicitly follows, minus the second-order term the donated Hessian
supplies. One `n x n` solve, and **zero function evaluations** — the model is
fitted entirely from points the simplex already holds. Its minimiser is a Newton
step from the best vertex, costing one evaluation per attempt:

```
x+ = v_0 - H^-1 g
```

This is the minimal modification that makes `H` enter *every* iteration rather
than only the initial simplex. When the curvature arrives as a CMA-ES
covariance it needs no inversion at all: `H^-1` is `sigma^2 C`, and
`InitialGeometry.solve` already applies it.

## 3. Four safeguards, each fixing a measured failure

The naive version of section 2 **loses, badly**. Each safeguard below was added
in response to a specific observed failure, not a hypothetical one. If you
change one, re-check the failure it was introduced for.

### 3.1 Geometry-preserving pivot, with a floor (`pivot_floor`, default 4.0)

*Failure:* replacing the **worst** vertex destroys the simplex. A short model
step drops the far vertex and pulls everything toward `v_0`, so the simplex
flattens onto a hyperplane and Nelder-Mead's own moves starve. Measured with
`B_0 = I` on the conditioned Ellipsoid: simplex volume was already **500x below
plain Nelder-Mead's by evaluation 200**, and the run finished **13 orders of
magnitude worse** — while consuming only 3 % of the budget. The damage was
*geometric, not budgetary.*

*Fix:* write the step in the edge basis, `x+ - v_0 = sum_j lambda_j d_j`.
Dropping vertex `j` multiplies the simplex volume by **exactly** `|lambda_j|`.
Take the largest `|lambda_j|` among the vertices the candidate beats (the
Lagrange-pivot rule from model-based DFO), **and decline the step entirely** when
even that best option falls below `pivot_floor`.

The floor is the part that matters, and its value carries a real finding.
Sweeping it (d=10, 5 seeds, 3000 evaluations, medians):

| `pivot_floor` | `I`     | `C_20`  | `C_320` | `H^-1`  |
|---------------|---------|---------|---------|---------|
| 0.1           | 1.2e+03 | 9.2e-01 | 4.5e-20 | 7.3e-23 |
| 1.0           | 3.1e-04 | 1.7e-01 | 3.6e-17 | 8.8e-18 |
| **4.0**       | 3.8e-09 | 2.5e-08 | 3.1e-15 | 5.1e-17 |
| 8.0           | 3.8e-09 | 2.5e-08 | 9.6e-14 | 3.0e-16 |
| plain NM      | 3.2e-10 | 9.6e-10 | 1.1e-09 | 2.4e-06 |

> **Short model steps are the harmful ones.** They are moves Nelder-Mead's own
> reflection already makes better, bought at the price of geometry. Long jumps —
> where a donated Hessian actually pays — sail past the floor. `1.0` means "may
> not shrink the simplex"; the default `4.0` means "must grow it".

### 3.2 A trust region that rides the simplex (`trust_low`, `trust_high`)

*Failure:* a free-floating trust radius collapses to zero and never recovers.

*Fix:* hold the radius at `trust_factor * (simplex extent in the H-metric)`, with
`trust_factor` adapted by the classic actual/predicted ratio inside a bounded
band. Anchoring to the simplex keeps the model step commensurate with
Nelder-Mead's own scale at every stage, with no problem-dependent constant — and
makes the trust ball inherit the donated ellipsoid's shape.

### 3.3 Ratio-gated schedule (`max_stride`, `ratio_threshold`)

*Failure:* on a landscape the curvature cannot describe, attempts burned 4.5 %
of the budget for nothing.

*Fix:* the interval between attempts doubles unless an attempt both improved the
incumbent **and** predicted the improvement (`ratio > ratio_threshold`). Gating
on predictiveness rather than improvement alone is what makes the overhead decay
toward zero: the run hands its budget back to plain Nelder-Mead, then wakes up if
the landscape turns quadratic again. On the Rippled Ellipsoid this settles at
0.9 % of budget with zero accepted steps.

### 3.4 Opportunity-cost gate (`gain_floor`, `gain_decay`)

*Failure:* the trust-region ratio asks whether the model predicted its *own*
decrease. A model fitted through `fit_scale` on a badly conditioned donated
matrix is perfectly self-consistent while pointing somewhere useless, so the
ratio gate stays open.

*Fix:* compare the model step's decrease-per-evaluation against a decaying
average of what the classic moves are achieving, and require at least
`gain_floor` of it. This asks the question that actually matters — *is this
evaluation better spent here?*

> Note: 3.4 was added before 3.1 and, on its own, did **not** fix the identity
> regression — because the damage was geometric, not budgetary. It is kept
> because it is the right guard for the budgetary failure mode, but 3.1 is what
> actually restored plain Nelder-Mead as the floor. Do not remove 3.1 on the
> assumption that 3.4 covers it.

## 4. Unknown magnitude (`fit_scale`)

A donated curvature is often known only up to a scalar: a CMA-ES covariance `C`
fixes the *shape* of `H^-1`, but its magnitude is absorbed into `sigma`. The
Newton step's length depends on that magnitude, so it matters.

Writing `H = alpha * H_1` with `H_1` normalised to unit geometric-mean
eigenvalue, the interpolation conditions stay **linear in `(g, alpha)`**:

```
g' d + alpha * (1/2 d' H_1 d) = f(x) - f_0
```

so one extra evaluated point makes the system square again in `n + 1` unknowns.
Nelder-Mead discards a point every iteration; `fit_scale=True` recycles those
discards as the extra rows and solves for `(g, alpha)` by least squares.

In the standalone PoC this turns a knife-edge into a flat line: with a fixed
scale the method is correct only at `alpha = 1`, and with the fit it is
insensitive across **six orders of magnitude** of magnitude error. See
`plots/neldermead_curvature/scale_robustness.png` after running the PoC.

## 5. The 2x2 arm design, and why the control exists

Comparing `neldermead | C_k` against `neldermead_hc | C_k` moves **two**
variables at once (simplex shaping off, model step on). The arms are therefore a
factorial over *how the conditioner reaches the run*:

|                      | no model step         | model step             |
|----------------------|-----------------------|------------------------|
| **isotropic simplex**| `neldermead_control`  | `neldermead_hc`        |
| **shaped simplex**   | `neldermead`          | `neldermead_hc_shaped` |

All four share `AlgorithmChoice.NELDERMEAD_HC` (except `neldermead`, which is the
original optimizer) and are separated by `LOCAL_VARIANTS` in
`experiments/conditioning/common.py`. A study must resolve its config through
`local_config_for(arm_key, ...)`, **not** `local_config(choice, ...)` — resolving
from the choice collapses every arm in a group onto one setting.

`neldermead_control` is handed the conditioner and ignores it. Two consequences:

1. It is the honest baseline — lean Nelder-Mead, same starting point, same
   rotation, same budget.
2. Its curves must **coincide across all seven conditioners**. That redundancy is
   a built-in check that the arms differ in nothing but the mechanism under test.
   It currently passes exactly (identical median/min/max).

`neldermead_control` with `model_step=False` is also bit-identical to
`NelderMeadOptimizer`, verified across 16 configurations. That is what makes the
comparison an ablation rather than two implementations.

## 6. Where this stands

### It works, spectacularly, with an accurate Hessian

exp1, Ellipsoid, unbounded, **25 seeds x d in {10, 20, 30}**, paired per seed
against that seed's own control run, in orders of magnitude below lean
Nelder-Mead:

| conditioner | simplex only | model step | both  |
|-------------|--------------|------------|-------|
| `C_20`      | +0.10        | -0.10      | -0.20 |
| `C_40`      | +0.14        | -0.16      | -0.21 |
| `C_80`      | +0.02        | -0.08      | -0.17 |
| `C_160`     | -0.17        | +0.04      | -0.01 |
| `C_320`     | -0.36        | **+0.76**  | +0.61 |
| `H^-1`      | -0.14        | **+22.82** | **+23.10** |
| `I`         | 0.00         | 0.00       | 0.00  |

### It does not (yet) pay off from a CMA-ES covariance

That table is the important one, and it is mostly a **negative result**. With the
true Hessian the model step is worth ~23 orders of magnitude. With actual CMA-ES
covariances at these snapshot depths it is worth **at most ~0.76 orders (about
6x)**, and below `C_160` it is slightly negative.

The natural reading: **the bottleneck is the estimator, not the consumer.** The
mechanism can clearly exploit accurate curvature; a CMA-ES covariance at
`k <= 32` iterations on this problem is not accurate enough to unlock it.

Two caveats on that table, both important:

- **`d = 10` saturates.** At `local_budget_per_dim = 500` lean Nelder-Mead
  already reaches `4.4e-20` on the 10-D Ellipsoid, so there is no headroom and
  every arm scores ~0. Pooling it in drags all the medians toward zero. The
  per-dimension panel of `aggregate_gain.png` shows this clearly — read it before
  quoting the pooled numbers.
- **One problem, one variant.** Ellipsoid only, `unbounded` only, `scaling=none`
  only.

### exp2 is a split result

On the CEC2017 F1/F3/F5 demo suite at `d = 10`, standalone `neldermead_hc` beats
standalone `neldermead` on the suite ECDF (AUC **0.281 vs 0.249**) using only
`B_0 = I` — the trust-region machinery earns its keep on its own. But every
CMA-ES hybrid comes out **below** the classic ones (best k: 0.269 vs 0.300).
Untested causes: probes run at loose `probe` tolerances on a short
2000-evaluation budget, the regime where a long model jump has least room to pay
off; and two of the three demo functions are not quadratic.

## 7. Open questions, roughly in priority order

1. **Is the estimator really the bottleneck?** Interpolate the donated
   eigenvalues toward the identity (`lambda -> lambda^beta`) and find where the
   payoff switches on. The PoC already has this sweep
   (`curvature_quality.png`); repeat it with CMA-ES covariances rather than a
   perturbed exact Hessian.
2. **Deeper CMA-ES snapshots.** The gain is still rising at `C_320`, the largest
   `k` measured. Does it keep climbing at `k = 64, 128`?
3. **Refresh the curvature mid-run.** Everything here uses one fixed `H`.
   Re-donating from a later snapshot, or updating from accepted model points, is
   the obvious answer to the stale-curvature case (Rosenbrock gains nothing
   because the Hessian at `x0` is stale everywhere along a curved valley).
4. **Why does `C_320 | both` have such high variance?** Median `3.0e-11` but
   min `3.4e-20`, max `3.4e-10`. Needs more seeds before anyone quotes it.
5. **The fitted `alpha` as a diagnostic.** It is a free, direct measure of how
   well a donated covariance matches local curvature — logged as
   `curvature_scale`, currently unexploited.
6. **Cost at scale.** The model step is `O(n^3)` per attempt (one `cond`, two
   solves). Irrelevant at `d <= 30`; measure at `d = 100`.
7. **exp2's hybrid regression** (section 6) is unexplained.

## 8. Where the code lives

| Path | What |
|---|---|
| `declivity/algorithms/neldermead_hc/neldermead_hc_optimizer.py` | the optimizer; the model step is one clearly marked block in `optimize()` |
| `declivity/algorithms/neldermead_hc/config.py` | every knob, each documented with the failure it guards |
| `declivity/logging/neldermead_hc_logger.py` | `model_attempts` / `model_ratio` / `trust_factor` / `curvature_scale` / `simplex_shape_quality` |
| `declivity/utils/initial_geometry.py` | `InitialGeometry.dense()` — added for the shape/magnitude split |
| `declivity/plotting/standard_panels.py` | panels, registered under the same semantic keys as Nelder-Mead so the two overlay |
| `experiments/conditioning/common.py` | `LOCAL_VARIANTS`, `local_config_for` — the 2x2 arms |
| `experiments/conditioning/export_nm_comparison.py` | re-cuts exp1 traces arm-wise; writes the aggregate |
| `experiments/neldermead_curvature/` | the standalone PoC: `hessian_completed.py`, `poc.py`, `validate.py` |

`experiments/neldermead_curvature/hessian_completed.py` is a **standalone
re-implementation** kept deliberately: it is where the idea is easiest to read,
and `validate.py` asserts its `model_step=False` mode is bit-identical to the
framework optimizer. It is not imported by the library.

## 9. Running it

```bash
# The ablation is honest: plain mode == framework Nelder-Mead, exactly.
PYTHONPATH=. uv run python experiments/neldermead_curvature/validate.py

# Standalone PoC: donated Hessian, no CMA-ES. Writes plots/neldermead_curvature/.
PYTHONPATH=. uv run python experiments/neldermead_curvature/poc.py \
    --dim 10 --seeds 21 --budget 2000 --only "ellipsoid 10d"

# One exp1 cell with all four arms (fast smoke test).
PYTHONPATH=. uv run python -m experiments.conditioning.exp1_hydra \
    experiment=demo objective=ellipsoid dim=10 variant=unbounded \
    num_workers=8 study_name=nmhc_demo

# The multi-dimension grid behind section 6. One cell at a time (the whole box
# goes to one cell, seeds parallelised across num_workers); ~1.5 h on 8 cores.
# Every stage is idempotent, so an interrupted run resumes where it stopped.
for D in 10 20 30; do
  PYTHONPATH=. uv run python -m experiments.conditioning.exp1_hydra \
      dim=$D variant=unbounded rotate=true \
      experiment=full objective=ellipsoid \
      study_name=nmhc_multi num_seeds=25 \
      'optimizers=[neldermead_control,neldermead,neldermead_hc,neldermead_hc_shaped]' \
      'snapshot_ks=[2,4,8,16,32]' 'scalings=[none]' \
      cmaes_evaluations_per_dim=1000 num_workers=8
done

# Re-cut those traces arm-wise and build the aggregate.
PYTHONPATH=. uv run python -m experiments.conditioning.export_nm_comparison \
    --study-name nmhc_multi --dim 10 20 30 --variant unbounded --num-seeds 25
```

A `-m` sweep over `dim` needs a launcher that runs cells sequentially;
`launcher=local` (the default group) starts every cell at once and
oversubscribes the box, so the loop above is the simplest correct form.

`cmaes_evaluations_per_dim=1000` is deliberate: `max_iterations` is derived as
`(cmaes_evaluations_per_dim * dim) // population_size`, and `snapshot_ks` only
needs `32 * dim` iterations, so the `full.yaml` default of 10000 runs CMA-ES far
longer than the snapshots require.

## 10. Gotchas for whoever picks this up

- **Resolve configs by arm key.** `local_config_for(key, ...)`, never
  `local_config(choice, ...)` — several arms share one `AlgorithmChoice`.
- **`neldermead_control` must stay flat across conditioners.** If it ever
  differs, an arm has picked up an unintended difference; fix that before
  reading any result.
- **Do not lower `pivot_floor` without re-running the identity conditioner.**
  It is the single guard standing between this and a 13-order regression.
- **`d = 10` at 500 evals/dim is saturated** on the Ellipsoid. Use `d >= 20` for
  anything you intend to quote.
- **The model step is not free on non-quadratics.** Expect ~1 % of budget as
  pure overhead when the curvature is uninformative. That is the designed
  behaviour, not a bug.
