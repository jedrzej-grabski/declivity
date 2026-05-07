# Empirical findings & gotchas

Distilled from the rotation study and CMA-ES handoff experiments.

## Initial Hessian on rotated Ellipsoid

Setup: rotated Ellipsoid with condition 10⁶, two regimes (10D/m=10,
50D/m=5), three B_0 choices (identity, Hessian diagonal, full Hessian).

**50D, m=5** (corrections cover 10 of 50 directions — decisive regime):

| Rotation | Identity | Diagonal | Full Hessian |
|---|---:|---:|---:|
| None | 6,423 | **24** | 24 |
| uniform_45 | 6,615 | 10,000+ | **24** |
| golden | 6,333 | 10,000+ | **24** |
| random | 7,274 | 10,000+ | **22** |

**10D, m=10** (corrections cover full space):

| Rotation | Identity | Diagonal | Full Hessian |
|---|---:|---:|---:|
| None | 292 | 22 | 22 |
| uniform_45 | 447 | 3,150 | 23 |
| golden | 431 | 1,457 | 23 |
| random | 422 | 2,022 | 99 |

**Key conclusions:**

1. **Wrong curvature is worse than no curvature.** Diagonal of a rotated
   Hessian is actively harmful — gives Hessian-direction info that's
   inconsistent with off-diagonal coupling. Worse than identity.

2. **Off-diagonal information is decisive when n >> 2m.** With n=50, m=5,
   corrections span 10 directions; the full matrix provides the remaining
   40 for free. Full Hessian is ~300x faster than identity here.

3. **When 2m ≥ n, B_0 matters less.** Corrections eventually span the full
   space and learn the curvature regardless of starting point.

## CMA-ES → L-BFGS-B handoff

Tested 6 transformations of CMA-ES's learned covariance C (with sigma σ)
as B_0 for L-BFGS-B:

| Transformation | Result | Why |
|---|---|---|
| Identity | Slow but converges | Baseline |
| Raw C | **Fails** (stuck at 1-26) | Wrong direction |
| C^{-1} | Converges ~10⁻⁶ | Correct direction |
| (σ²C)^{-1} | Converges ~10⁻⁶ | Correct direction + scale |
| C/tr(C)*n | **Fails** (stuck at 1-34) | Wrong direction |
| (C/tr(C)*n)^{-1} | Converges ~10⁻⁶ | Correct direction, neutral scale |

**Critical insight: L-BFGS-B (bounded) maintains B (Hessian), not B^{-1}
(inverse Hessian).** The B is needed for the Cauchy point — walking the
projected gradient and evaluating the quadratic model `g'd + ½d'Bd` at
breakpoints.

CMA-ES covariance C is proportional to **B^{-1}** (large variance = flat =
small curvature). So passing C directly is backwards — it tells L-BFGS-B
the steep directions are flat and vice versa.

If handing off to standard L-BFGS (unconstrained), which stores H ≈ B^{-1},
passing C directly **would** be correct. The distinction is critical.

**Scale doesn't matter, direction does.** All three inverses perform
nearly identically. The theta parameter adapts the global scale within a
few iterations, so initial scaling is irrelevant. Only the directional
structure of the matrix matters.

## Cost analysis

DENSE B_0 adds O(m n²) per iteration (Cauchy point matrix-vector products,
subspace min Cholesky solves on free subblock, correction-pair updates).
For n ~ 100 this is acceptable; for large-scale problems it defeats the
limited-memory property.

DIAGONAL B_0 is O(m n) — same complexity as identity, no real overhead.

## Gotchas

### Symmetric starting points

`x0 = np.full(n, c)` (all coordinates equal) can accidentally align with
the eigenvectors of structured rotations like `uniform_45`. This causes
L-BFGS-B to converge in 4 evals from gradient direction alone — the
**function value plot looks like a miracle but is an artifact of x0**.

Fix: use random uniform initialization. The rotation_study script does
`rng = np.random.default_rng(42); x0 = rng.uniform(-100, 100, size=n)`.

### Stall guard

The optimizer has branches (direction below epsilon, line search failure)
that `continue` without consuming an evaluation. Without a counter, this
loops forever. Current guard: `max_consecutive_resets = 20`.

If you see runs terminating in <30 evals on hard problems with message
"Stalled: repeated memory resets", investigate before trusting the result.

### CMA-ES partial-population evaluations

CMA-ES counts +1 per generation for the mean evaluation in addition to
the population. So `evals_per_generation = pop_size + 1`. Get this right
when computing handoff offsets.

### Iteration vs evaluation axes for handoff plots

CMA-ES iterations and L-BFGS-B iterations are not comparable (different
work per iteration). When prepending CMA-ES history to L-BFGS-B logs,
**offset L-BFGS-B iterations by `len(cmaes_log.iteration)`** so the
iteration axis is continuous, not restarting.

### Polish characters in markdown

The Edit tool sometimes fails to match strings containing Polish diacritics
(ż, ć, ł, etc.) due to encoding. Use Write for whole-file replacements when
this happens, or stick to ASCII in old_string anchors.

### Curvature too sharp

If `initial_hessian` is set to the median of true diagonal curvature on a
problem with high variance (e.g. ~2600 on Ellipsoid with cond 10⁶), the
optimizer can cycle between too-small steps and memory resets. The stall
guard catches this but the result is meaningless. Use full Hessian or
identity instead.

### NumPy overflow warnings in matmul

50D rotation chains can trigger `RuntimeWarning: overflow encountered in
matmul` during construction. Wrapped with `np.errstate(over="ignore",
invalid="ignore", divide="ignore")` in `RotatedEllipsoid` and
`InitialHessian` operations. The values are valid; numpy is just paranoid
about intermediates.
