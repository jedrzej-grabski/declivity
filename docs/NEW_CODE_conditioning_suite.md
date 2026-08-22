# NEW CODE: The conditioning experiment suite

`experiments/conditioning/` is a pair of staged, cache-aware studies of one
question: *what does a CMA-ES-learned geometry buy a local optimizer, and
when should you switch?* This document records the new framework capabilities
that carry them and the shape of the pipelines, so the next study can reuse
both.

## What was added to `declivity/`

| Piece | Where | What it does |
|---|---|---|
| `CMAESSnapshot` / `CMAESPath` / `record_cmaes_path` / `save_cmaes_path` / `load_cmaes_path` | `benchmarking/cmaes_path.py` | Run CMA-ES once, snapshot the **full resumable state** (mean, σ, C, both paths, cached `(B, D)`, funhist, incumbent) every `interval` generations, persist to `trace.json` + `snapshots.npz` + `meta.json`. Advances through the `CMAESState` resume machinery with a shared RNG: verified byte-identical to a continuous run (mean/σ/C/pc/ps all `==` after 40 generations). `CMAESSnapshot.to_state()` restores a resumable `CMAESState`. |
| `snapshot_geometry` / `local_seeding_kwargs` / `run_conditioned_local` / `ConditionedLocalAlgorithm` | `benchmarking/conditioning.py` | Turn a snapshot (or any curvature source) into per-optimizer ctor kwargs and run a conditioned local. Powell receives the covariance eigenvectors **scaled by `sqrt` of the eigenvalues** (normalised to the longest, rank-floored at `1e-6`): direction magnitudes set the line search's first bracket, so unit vectors would discard the learned scale. Nelder-Mead gets the geometry plus an explicit `simplex_base_size` (mandatory for unbounded problems, where the bounds-derived default is infinite). |
| `compose_switch_trace` | `benchmarking/conditioning.py` | Offline interleaved-hybrid composition: CMA-ES timeline + probes spliced in at their switch evaluations, running-min, CMA-ES tail shifted right by each probe's cost. The offline equivalent of `InterleavedCMAESLBFGSB` without feedback: so **every k shares one CMA-ES run and one probe set**. |
| `ProblemFamily` | `benchmarking/problem.py` (+ a resolve hook in `Benchmark._build_jobs`) | Per-seed problem *instances* (same statistical problem, a fresh rotation/shift per seed) flowing through the standard `Benchmark` loop under one problem name. |
| `plot_suite_ecdf` | `plotting/benchmark.py` | COCO-style aggregated ECDF: per-problem target grids (gap to each problem's own `f*`), every (problem, target) pair weighing equally, one curve per algorithm. |
| `numerical_hessian` / `spd_regularize` | `utils/hessian.py` | Central-difference Hessian (`2n²+2n+1` evals; exact for quadratics like CEC 2017 F1) + eigenvalue-floor SPD projection. |
| `CMAESPath.snapshot_at_or_before` | `benchmarking/cmaes_path.py` | Resolves a requested `k·d` to the latest recorded state. A run that converged sooner has a frozen state, so its terminal snapshot *is* the state at `k·d`; each run's `config.yaml` records both the requested and the used iteration. |
| `xmax=` on `plot_convergence_overlay`, `show_subtitle=` on `plot_suite_ecdf`, 300 dpi default in `save_if_path` | `plotting/` | Axis clipping, less chrome, sharper output. |
| Final-state accessors | the four local optimizers | `BFGSOptimizer.final_inverse_hessian`, `PowellOptimizer.final_directions`, `NelderMeadOptimizer.final_simplex`, `LBFGSBOptimizer.final_corrections()`: the learned state at the end of the last `optimize()`, for persistence. |

## The artifact store

Each study writes under `results/conditioning/<exp>/<study-name>/` (data) and
`plots/conditioning/<exp>/<study-name>/` (figures):

```
study.yaml                          full spec dump
setup/dDDD/seedSS/                  x0.npy, rotation.npy, meta.yaml
cmaes/<variant>/dDDD[/fFF]/seedSS/  trace.json, snapshots.npz, meta.json, config.yaml
hessian/dDDD/seedSS/                hessian.npy, meta.yaml           (exp1)
local/.../seedSS[/itIIIIII|alone]/  run.npz (trace + x_best path + final state), config.yaml
benchmarks/...                      Benchmark / composed traces.json  (what --replot reads)
```

Every stage is idempotent: it loads what exists and only computes what is
missing (`--force-cmaes` / `--force-probes` override; `--skip-cmaes` forbids
running the expensive stage and fails loudly if artifacts are missing). This
is what makes the suite remote-friendly: stages can run on different machines
against a synced store, and figures re-render anywhere via `--replot`.

## Experiment 1: conditioners (`exp1_conditioners.py`)

Same optimizer, same `x0`, different initial geometry: `C` after `k·d` CMA-ES
iterations (`INVERSE` transform), the FD true-Hessian `H⁻¹`, and `I`. Per-seed
random rotations compose with CEC 2017 F1's internal transform; bounded and
unbounded variants share seeds and starting points. Note: under a per-seed
rotation the F1 optimum can leave the `[-100, 100]^d` box, so the *bounded*
variant's reachable optimum sits on the boundary and the gap plateaus above
0 by design; that is the boundary-handling regime (direction projection /
feasible-step capping through the `ConstraintHandler`, never orthogonal
clipping of trial points along the search path).

### `--objective {cec,ellipsoid}`

`build_family()` in `common.py` selects the objective:

- `cec` (default): CEC2017 F1 ("Bent Cigar") via `--edition`/`--function`, as
  above. CEC problems carry their own internal fixed rotation/shift baked
  into the compiled evaluator, *on top of* whatever `--no-rotate` applies
  here — so "unrotated CEC F1" is never actually axis-aligned.
- `ellipsoid`: `declivity.utils.benchmark_functions.Ellipsoid`, the
  axis-aligned, `10^6`-conditioned quadratic
  `f(x) = sum_i 10^(6 i/(d-1)) x_i^2`. It carries **no** internal
  rotation/shift of its own, so `--no-rotate` makes it genuinely axis-aligned
  (diagonal Hessian) and the default random rotation is the *only* source of
  coordinate coupling — the canonical CMA-ES-covariance-converges-to-Hessian
  test function. Bounds default to `[-100, 100]`, matching the suite's CEC
  sampling box, so both bounded and unbounded variants work unchanged.
  `--edition`/`--function` are not applicable and are rejected (not silently
  ignored) when combined with `--objective ellipsoid`.

Use `ellipsoid` when you want a controlled, analytically-understood
conditioning study (known diagonal Hessian, exact gradient, no confound from
CEC's own baked-in transform); use `cec` for the suite's original CEC2017 F1
study or to compare against other CEC problems.

```bash
PYTHONPATH=. uv run python experiments/conditioning/exp1_conditioners.py --demo --objective ellipsoid
PYTHONPATH=. uv run python experiments/conditioning/exp1_conditioners.py --demo --objective ellipsoid --no-rotate
```

Demo findings (d=10, unbounded, k ∈ {2,4,8,16,32}): BFGS with `H⁻¹` reaches
the `1e-9` gap in ~170 evaluations, the `C_kd` conditioners cluster ~480–630
(non-monotone in k: the mid-range `C_80` is fastest, and by `C_320` CMA-ES has
converged so far that the collapsed covariance is worse than identity), and
identity lands ~640. Powell needs `C_320` or `H⁻¹` to crack the
10⁶-conditioned rotated cigar; a needle-shaped `H⁻¹` simplex (axis ratios at
the `1e-3` floor) actively *hurts* Nelder-Mead.

## Experiment 2: full hybrid (`exp2_hybrid.py`)

One recorded CMA-ES path per (function, seed) at snapshot granularity
`g·d` iterations; one probe per (snapshot, optimizer) from the CMA-ES mean
with the covariance geometry handed off; hybrid contenders "CMA+opt, k"
composed offline for every `k·d`, plus the standalone local and the CMA-ES
curve itself. ECDFs are computed from the composed convergence curves.

Defaults mirror the CMABFGS reference (λ = 4d, k ∈ {0.5, 1, 2, 4, 8},
25·d² CMA-ES iterations); everything is a spec field / CLI flag.

## Running

```bash
# demos (small, local)
PYTHONPATH=. uv run python experiments/conditioning/exp1_conditioners.py --demo
PYTHONPATH=. uv run python experiments/conditioning/exp2_hybrid.py --demo

# full scale
PYTHONPATH=. uv run python experiments/conditioning/exp1_conditioners.py --num-workers 8
PYTHONPATH=. uv run python experiments/conditioning/exp2_hybrid.py --num-workers 8

# figures only, from the store
... exp1_conditioners.py --demo --replot
```
