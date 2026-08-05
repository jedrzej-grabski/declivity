# L-BFGS-B Initial Hessian: Implementation, Experiments, and Results

*Report for thesis supervision meeting — Jedrzej Grabski*

---

## 1. Starting point: the Woodbury inconsistency

The standard L-BFGS-B algorithm uses the compact Hessian representation:

    B = theta * I - W * M * W'

where `theta` is a scalar updated each iteration, `W` contains the correction
vectors, and `M` is a small matrix derived from inner products of the stored
`(s, y)` pairs. The key observation: the identity matrix `I` appears everywhere
as the "base" Hessian. When we allow a user-provided `B_0` in place of `I`,
it must be threaded through every component:

| Component | Original (B_0 = I) | Generalized |
|---|---|---|
| W matrix | `[Y \| theta * S]` | `[Y \| theta * B_0 * S]` |
| M^{-1} block | `theta * S'S` | `theta * S' B_0 S` |
| T matrix | `theta * S'S + L D^{-1} L'` | `theta * S' B_0 S + L D^{-1} L'` |
| Woodbury solve | `(1/theta) * I` | `(1/theta) * B_0^{-1}` |

In our first implementation, we only used `B_0` in the Cauchy point (active
set identification) but left the subspace minimization using the original `I`.
This was inconsistent — the Cauchy point saw one Hessian, the Newton step saw
another. We corrected this by threading `B_0` through the entire algorithm.

## 2. Cost analysis: when is a full matrix affordable?

Replacing `I` with a general `B_0` changes the per-iteration cost:

| B_0 type | Cost per iteration | Storage | B_0 * v |
|---|---|---|---|
| Identity | O(m n) | 0 | free |
| Diagonal | O(m n) | O(n) | O(n) element-wise |
| Full matrix | O(m n^2) | O(n^2) + Cholesky | O(n^2) matvec |

For the diagonal case, every `B_0` operation is element-wise multiplication,
so the asymptotic cost is unchanged from the identity case. For a full matrix,
each `B_0 * v` product costs O(n^2), and the `S' B_0 S` Gram matrix costs
O(m n^2). The Cholesky factorization of `B_0` is computed once at initialization
(O(n^3)) and reused for all subsequent `B_0^{-1}` solves (O(n^2) each).

For our thesis-scale problems (n = 10 to 50), the full matrix cost is entirely
negligible. For n = 10, an iteration with dense `B_0` costs ~1,000 multiplications.
For n = 10,000 (typical industrial L-BFGS-B use), it would cost 10^9 — defeating
the purpose of limited-memory. Our implementation cleanly branches via an
`InitialHessian` helper class with an `InitialHessianMode` enum (`DIAGONAL`
or `DENSE`), so the optimizer code calls the same interface regardless of the
type provided.

## 3. Rotated test problems

To test whether off-diagonal curvature information actually helps, we need
problems where the Hessian is non-diagonal. The standard Ellipsoid has a
diagonal Hessian (eigenvalues from 2 to 2×10^6, condition number 10^6). We
introduced `RotatedEllipsoid`, which applies an orthogonal rotation `R` to the
input: `f(x) = Ellipsoid(R x)`. The Hessian becomes `H = 2 R' diag(scales) R`
— a full matrix with off-diagonal terms.

Four rotation modes with increasing "scatter" of the Hessian:

| Mode | Description | Diagonal fraction |
|---|---|---|
| None | Axis-aligned (original Ellipsoid) | 1.00 |
| Uniform 45° | Chain of 45° Givens rotations in consecutive planes | 0.49 (10D), 0.91 (50D) |
| Golden angle | Chain of golden-angle (137.5° × k) rotations | 0.83 (10D), 0.94 (50D) |
| Random | QR decomposition of a random matrix | 0.26 (10D), 0.18 (50D) |

The "diagonal fraction" measures how much of the Hessian's total energy
(Frobenius norm squared) lives on the diagonal. Lower values mean more
off-diagonal structure — more information that a diagonal approximation misses.

The contour plots of the (x1, x2) slice show the effect visually. The arrows
are the principal curvature directions (Hessian eigenvectors projected onto the
plotted plane):

![Landscape grid](../plots/lbfgsb/rotation_study/landscapes.png)

As the rotation increases, the contour ellipses tilt away from the coordinate
axes. A diagonal Hessian approximation can only scale along the coordinate
axes — it cannot represent the tilted curvature.

## 4. Experiment 1: Full Hessian vs Diagonal vs Identity

We tested three initial Hessian choices on all four rotations in two regimes:

- **n = 10, m = 10**: corrections can span 2m = 20 directions, which fully
  covers 10D. The initial Hessian matters less because corrections learn fast.
- **n = 50, m = 5**: corrections span only 2m = 10 of 50 directions. The
  remaining 40 directions are determined entirely by `B_0`.

### Results: n = 50, m = 5 (the decisive case)

![50D summary](../plots/lbfgsb/rotation_study/50D_m5_summary.png)

| Rotation | Identity | Diagonal | Full Hessian |
|---|---|---|---|
| None (axis-aligned) | 5,064 | **23** | 23 |
| Uniform 45° | 3,508 | 10,000+ | **22** |
| Golden angle | 5,968 | 10,000+ | **22** |
| Random | 6,111 | 10,000+ | **23** |

The full Hessian converges in 22-23 evaluations regardless of the rotation.
It is completely insensitive to the coordinate system because it provides
the correct curvature in all 50 directions from the start.

The diagonal is excellent on the axis-aligned Ellipsoid (23 evals, matching
the full matrix) but catastrophic on all rotated problems (budget exhausted
at 10,000 evals, function values still in the 10^4 to 10^6 range). This is
because the diagonal of a rotated Hessian captures per-coordinate curvature,
which is a misleading mixture of eigenvalues. The algorithm fights this
wrong prior and makes almost no progress.

The identity is a middle ground: unbiased (no wrong information), but slow
(3,500-6,100 evals) because it provides no curvature information at all.

### Results: n = 10, m = 10

![10D summary](../plots/lbfgsb/rotation_study/10D_m10_summary.png)

With corrections spanning the full space, the full Hessian still wins (18-21
evals) but the identity also converges reasonably (388-571 evals on most
rotations). The diagonal remains catastrophic on rotated problems (546-2,389
evals with poor precision).

### Convergence curve example: Random rotation, 50D

![Random 50D convergence](../plots/lbfgsb/rotation_study/50D_m5_random_convergence.png)

The full Hessian (green) drops to 10^{-13} immediately. The identity (red)
converges gradually over thousands of evaluations. The diagonal (orange) is
flat — stuck, making no meaningful progress.

## 5. Experiment 2: CMA-ES covariance handoff

CMA-ES learns a covariance matrix C that encodes the search distribution's
shape. Large variance in a direction means the function is flat there; small
variance means it is steep. This is the inverse relationship to the Hessian
(large Hessian = steep = small variance).

After running CMA-ES for a warm-up phase, we extracted the learned covariance
matrix and sigma from the diagnostic logs, then transformed them into initial
Hessian matrices for L-BFGS-B. We tested six transformations:

| Transformation | Formula | Rationale |
|---|---|---|
| Identity | I | Baseline: no CMA-ES info used |
| Raw covariance | C | Wrong direction: large where Hessian is small |
| Inverse | C^{-1} | Correct direction: inverts the covariance-Hessian relationship |
| Inverse scaled | (sigma^2 C)^{-1} | Correct direction and absolute scale |
| Normalized | C / tr(C) * n | Unit-trace normalization, wrong direction |
| Inv. normalized | (C / tr(C) * n)^{-1} | Correct direction with neutral scale |

Note on the convergence plots: the dashed vertical line marks the CMA-ES to
L-BFGS-B handoff point. All curves overlap before the handoff (same CMA-ES
run) and diverge after it depending on the transformation used.

### Results: 50D, m=5, 300 CMA-ES generations

![Uniform 45 convergence](../plots/hybrid/handoff_study/uniform_45_50d_convergence.png)

![Random convergence](../plots/hybrid/handoff_study/random_50d_convergence.png)

### Results: 10D, m=5, 100 CMA-ES generations

![Uniform 45 10D convergence](../plots/hybrid/handoff_study/uniform_45_10d_convergence.png)

![Random 10D convergence](../plots/hybrid/handoff_study/random_10d_convergence.png)

### Consistent pattern across all rotations (50D):

| Transformation | Typical final f(x) | L-BFGS-B evals | Converged? |
|---|---|---|---|
| **Inverse C^{-1}** | **~10^{-6}** | ~4,500 | Yes |
| **Inverse scaled (s^2 C)^{-1}** | **~10^{-6}** | ~4,400 | Yes |
| **Inv. normalized (C/tr*n)^{-1}** | **~10^{-6}** | ~4,500 | Yes |
| Identity | ~10^{-3} to 10^{-2} | 5,000 (budget) | No |
| Raw covariance C | ~1 to 26 | 5,000 (budget) | No |
| Normalized C/tr(C)*n | ~1 to 34 | 5,000 (budget) | No |

The three inverses are the clear winners. They correctly convert the
covariance's "spread" into the Hessian's "curvature", allowing L-BFGS-B
to benefit from the curvature structure that CMA-ES learned during its
warm-up phase.

Raw covariance and normalized covariance are actively harmful — they have
the Hessian-covariance relationship backwards (large where the Hessian
should be small), giving the optimizer worse information than no information
at all.

The three inverses perform similarly, suggesting that the directional
information (which directions are steep/flat) matters more than the absolute
scale. This makes sense: the L-BFGS-B theta parameter adapts the overall
scale each iteration anyway, so the initial absolute scale is quickly
corrected.

In 10D the results are noisier because m=5 corrections span 10 directions
(the full space), so B_0 matters less and noise from CMA-ES covariance
quality dominates. The 50D regime (where corrections cover only 10 of 50
directions) is the decisive case for the handoff.

## 6. Key observations

1. **Wrong curvature is worse than no curvature.** Both the diagonal on
   rotated problems and the raw covariance are worse than identity. The
   algorithm's L-BFGS corrections have to fight against the misleading
   prior rather than build on a neutral starting point.

2. **Off-diagonal information is decisive when n >> 2m.** In 50D with m=5,
   the corrections can only learn 10 directions. The full matrix provides
   the remaining 40 for free. This is the regime where the full matrix
   is 160x faster than identity.

3. **Inversion is necessary for the CMA-ES handoff.** The covariance is
   inversely related to the Hessian. Passing it directly is counterproductive.
   The inverse (with or without sigma scaling) is the correct transformation.

4. **Scaling has minimal impact.** C^{-1}, (sigma^2 C)^{-1}, and
   (C/tr(C)*n)^{-1} all perform nearly identically because theta adapts
   the overall scale within a few iterations. Only the directional
   structure of the matrix matters.

5. **The implementation cost is O(m n^2) per iteration with a dense B_0.**
   This is acceptable up to n ~ 100 but would defeat the limited-memory
   property for large-scale problems.

## 7. Implementation summary

Files modified or created:

| File | Change |
|---|---|
| `declivity/algorithms/lbfgsb/initial_hessian.py` | New: `InitialHessian` class with `InitialHessianMode` enum, dispatches diagonal vs dense operations |
| `declivity/algorithms/lbfgsb/lbfgsb_optimizer.py` | Rewrote to use `InitialHessian` throughout: Cauchy point, W matrix, M matrix, T matrix, Woodbury solve |
| `declivity/algorithms/lbfgsb/config.py` | Added `initial_hessian` (None/scalar/vector/matrix) and `persist_initial_hessian` flag |
| `declivity/algorithms/cmaes/cmaes_optimizer.py` | Added `get_learned_covariance()`, `sigma`, `mean` public API |
| `declivity/logging/cmaes_logger.py` | Now stores the full covariance matrix; `diag_covariance_matrix` flag for every-generation storage |
| `declivity/utils/benchmark_functions.py` | New: `RotatedEllipsoid` with four rotation modes (uniform_45, golden, random, custom) |
| `declivity/plotting/multi_algorithm_plotter.py` | New: `plot_function_landscape()`, `plot_function_landscape_grid()`, `plot_matrix_diagonal_comparison()`, `plot_labeled_convergence_comparison()`, `plot_evaluation_bar_chart()` |

Benchmarks:

| File | What it tests |
|---|---|
| `examples/lbfgsb_rotation_study.py` | Full Hessian vs diagonal vs identity across 4 rotations and 2 dimensionality regimes |
| `examples/cmaes_handoff_study.py` | 6 covariance transformations for the CMA-ES -> L-BFGS-B handoff across all rotations |

## 8. Next steps

- Test on non-quadratic functions (Rosenbrock, CEC benchmarks) where the
  Hessian changes across the landscape and line search accuracy matters more.
- Explore how many CMA-ES generations are needed for the covariance to be
  useful — the tradeoff between warm-up cost and L-BFGS-B acceleration.
- Compare the hybrid CMA-ES -> L-BFGS-B strategy against standalone CMA-ES
  and standalone L-BFGS-B on the same total evaluation budget.
