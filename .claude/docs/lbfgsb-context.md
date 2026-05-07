# L-BFGS-B implementation context

State as of branch `lbfgs-implementation`. Read this before working on the
L-BFGS-B optimizer or related benchmarks.

## What's implemented

A pure-Python port of L-BFGS-B v3.0 (Morales-Nocedal 2011), integrated into
the thesis framework via `AlgorithmFactory.create_optimizer(AlgorithmChoice.LBFGSB, ...)`.

Includes the full Fortran algorithm: generalized Cauchy point, subspace
minimization via Woodbury identity, More-Thuente line search (port of
`dcsrch`/`dcstep`), and the projected Newton safeguard.

## Initial Hessian abstraction

`src/algorithms/lbfgsb/initial_hessian.py` — `InitialHessian` class with
`InitialHessianMode` enum (`DIAGONAL`, `DENSE`).

Config accepts `initial_hessian: None | float | NDArray`:
- `None` → identity (theta * I), updated by ys/yy ratio
- scalar → uniform diagonal
- 1D array → arbitrary diagonal
- 2D array → full Hessian (DENSE mode, precomputes Cholesky)

`persist_initial_hessian: bool` — if True, theta is locked at 1.0 and B_0
stays as supplied; if False, theta adapts each iteration.

B_0 is threaded through **all five** algorithm components — this was a real
bug previously: Cauchy point used B_0 but subspace min implicitly used I.
Now consistent in: Cauchy point, W projection, M^{-1}, T matrix, Woodbury solve.

For DENSE mode, subspace min extracts the free-variable subblock of B_0 and
uses `cho_solve` on it. Cost is O(m n²) per iteration (vs O(m n) for diagonal),
which is why dense should not be used for large n.

## Line search

`LineSearchMethod` enum: `MORE_THUENTE` (default, robust) or `ARMIJO`
(simple backtracking with `contraction_factor`). Configurable via
`LBFGSBConfig`.

## Stall guard

`max_consecutive_resets = 20` in the optimizer. Prevents infinite loops when
direction is below machine epsilon or line search fails — these branches
don't consume evaluations, so without the guard the optimizer can spin
forever. Tuned to 20; lower values (e.g. 5) cause spurious early termination
on edge cases.

## Logging

`LBFGSBLogData`: `function_value`, `gradient_norm`, `projected_gradient_norm`,
`step_length`, `theta`, `num_free_vars`, `num_corrections`,
`line_search_iters`. All gated by `diag_*` flags on the config.

## CMA-ES additions for handoff

`src/algorithms/cmaes/cmaes_optimizer.py` exposes:
- `get_learned_covariance()` → `CovarianceMatrix`
- `sigma` property
- `mean` property

`CMAESLogData.covariance_matrix` field added. With
`diag_covariance_matrix=True` it stores every generation; False stores only
the latest (replaces in-place).

## Rotated test functions

`src/utils/benchmark_functions.py` — `RotatedEllipsoid` with four modes:
- `none` — axis-aligned (use `Ellipsoid` instead)
- `uniform_45` — Givens chain in planes (i,i+1) at 45°
- `golden` — Givens chain at (k+1) * 137.5° per plane (aperiodic)
- `random` — QR factor of N(0,1) matrix (full random orthogonal)

Provides `gradient(x)`, `hessian` property, `hessian_diagonal` property.
Hessian = 2 * R' @ diag(scales) @ R.

Diagonal energy fractions (||diag||²/||H||²):

| Rotation | 10D | 50D |
|---|---:|---:|
| None | 1.00 | 1.00 |
| uniform_45 | 0.49 | 0.91 |
| golden | 0.83 | 0.94 |
| random | 0.26 | 0.18 |

## Plotting

`MultiAlgorithmPlotter.plot_labeled_convergence_comparison()` — 4-panel
(by evals, by iter, projected gradient norm, step length). Accepts:
- `handoff_eval` — vertical dashed line on eval-based panels
- `handoff_iter` — vertical dashed line on iter-based panels

Other useful methods: `plot_evaluation_bar_chart`, `plot_function_landscape`,
`plot_function_landscape_grid`, `plot_matrix_diagonal_comparison`.

## Benchmarks

- `examples/lbfgsb_benchmark.py` — Sphere: L-BFGS-B vs CMA-ES
- `examples/lbfgsb_hessian_benchmark.py` — Ellipsoid with various B_0 choices
- `examples/lbfgsb_rotation_study.py` — full Hessian vs diagonal vs identity
  across 4 rotations × 2 regimes (10D/m=10, 50D/m=5)
- `examples/cmaes_handoff_study.py` — 6 covariance transformations for
  CMA-ES → L-BFGS-B handoff, prepends full CMA-ES history to plots
- `examples/cmaes_to_lbfgsb_benchmark.py` — earlier handoff experiment

Run with `PYTHONPATH=. pdm run python examples/<name>.py`.

`cmaes_handoff_study.py` accepts CLI args: `dimensions memory_size cmaes_gens`
(defaults: 50 5 300). For 10D use `10 5 100` — more gens converges CMA-ES
fully and L-BFGS-B has nothing to do.

## Documentation

- `docs/lbfgsb_lecture.md` — full algorithm lecture (11 sections)
- `docs/lbfgsb_initial_hessian_design.md` — cost analysis, dispatching strategy
- `docs/supervisor_report_initial_hessian.md` — supervisor report w/ graphs
- `docs/wnioski.md` — Polish-language summary for supervisor (sections 1-5)
- `notes/benchmark_initial_hessian.md` — Polish benchmark analysis
- `notes/research_initial_hessian_rotated.md` — rotated problem research

## Key file map

```
src/algorithms/lbfgsb/
  __init__.py             exports LBFGSBConfig, LBFGSBOptimizer, LineSearchMethod, InitialHessian, InitialHessianMode
  config.py               LBFGSBConfig dataclass
  lbfgsb_optimizer.py     ~900 lines, the algorithm
  initial_hessian.py      InitialHessian class with multiply/solve/quadratic_form/etc.
  line_search.py          More-Thuente + Armijo

src/logging/lbfgsb_logger.py
src/algorithms/cmaes/cmaes_optimizer.py    public API additions
src/logging/cmaes_logger.py                covariance_matrix logging
src/utils/benchmark_functions.py           RotatedEllipsoid
src/plotting/multi_algorithm_plotter.py    convergence/landscape/diagonal plots
```
