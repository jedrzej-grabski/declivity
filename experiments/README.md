# Experiments

Runnable studies that exercise the library. Grouped by topic; the
`plots/` tree mirrors this layout — every experiment writes its outputs
into the corresponding subdirectory.

Run any experiment as a module from the project root:

```bash
PYTHONPATH=. pdm run python -m experiments.<group>.<name>
# or directly
PYTHONPATH=. pdm run python experiments/<group>/<name>.py [--flags ...]
```

The two short-form PDM scripts cover the canonical starting points:

```bash
pdm run run-example      # experiments/basic/simple_optimization.py
pdm run run-r            # experiments/cross_validation/des_vs_r.py
```

**Writing a new experiment?** See the step-by-step guide
[`docs/NEW_CODE_building_an_experiment.md`](../docs/NEW_CODE_building_an_experiment.md)
— the conventions (Problem → algorithm specs → `Benchmark` → declarative
plots), where custom code belongs, and the anti-patterns to avoid.

## `basic/` — tutorial demos and sanity checks

| Script                       | What it does                                                    | Output                                  |
|------------------------------|-----------------------------------------------------------------|-----------------------------------------|
| `simple_optimization.py`     | Runs every algorithm on a 10D Sphere and dumps each one's default diagnostic-panel view | `plots/basic/simple_optimization/`      |
| `covariance_adaptation.py`   | CMA-ES on Sphere / Ellipsoid; visualizes empirical covariance evolution (eigenvalues, condition, 2D ellipse snapshots) | `plots/basic/covariance_adaptation/`    |
| `declarative_plotting.py`    | End-to-end demo of `plot_metrics` and `plot_comparison` — including `PanelRegistry.common([algos])` introspection | `plots/basic/declarative_plotting/`     |
| `declarative_benchmark.py`   | End-to-end demo of `plot_benchmark_convergence` and `plot_benchmark_boxplot` with a real CMA-ES → L-BFGS-B handoff | `plots/basic/declarative_benchmark/`    |
| `custom_handoff.py`          | Builds a DES → L-BFGS-B handoff from scratch by subclassing `HandoffAlgorithm` (one `run_phases()` method) | `plots/basic/custom_handoff/`           |
| `custom_algorithm.py`        | Multi-start CMA-ES via `BenchmarkAlgorithm` directly; shows the generic extension point | `plots/basic/custom_algorithm/`         |
| `plotter_showcase.py`        | Three one-call figures proving the declarative plotter needs no per-algorithm code (cross-family overlay, multi-seed median+IQR, all-diagnostics grid). Write-up: `docs/NEW_CODE_Plotter_showcase.md` | `plots/basic/plotter_showcase/`         |
| `benchmark_showcase.py`      | One `Benchmark` grid over four heterogeneous runners (every rung of the extension hierarchy) → multi-seed convergence (median + IQR); demos same-seed fairness + `traces.json` persistence. Write-up: `docs/NEW_CODE_Benchmark_showcase.md` | `plots/basic/benchmark_showcase/`       |
| `constrained_rosenbrock.py`  | 2D Rosenbrock under box-only vs. box + custom-penalty disk inequality across DES, CMA-ES, L-BFGS-B; demos the `ConstraintHandler` API end-to-end | `plots/basic/constrained_rosenbrock/`   |

## `cross_validation/` — checks against external references

| Script                  | What it does                                                                          | Output                                              |
|-------------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------|
| `des_vs_r.py`           | Runs DES on CEC2017 F10 (10D, 10 seeds, fixed `x0=50·1`); writes convergence + summary CSVs that can be diffed against output from `reference/cmaes.r` / `reference/mf_cmaes.r` | `reference/outputs/python_*_f10_d10.csv`            |
| `cmaes_vs_reference.py` | Convergence-equivalence oracle between framework CMA-ES and the historical reference port across Sphere / Ellipsoid / Rosenbrock / Rastrigin / Ackley / CEC17 F10; produces convergence overlays, state trajectories, and a max-diff heatmap | `plots/cross_validation/cmaes_vs_reference/`        |
| `cmaes_components.py`   | Framework CMA-ES under different `RepairStrategy` / `PopulationInitializer` injections (default `LamarckianRepair` vs `IdentityRepair` vs `NormalPopulationInitializer`) — confirms both seams are live and the default is non-regressing | `plots/cross_validation/cmaes_components/`          |

## `lbfgsb/` — L-BFGS-B feature studies

| Script               | What it does                                                                              | Output                              |
|----------------------|-------------------------------------------------------------------------------------------|-------------------------------------|
| `sphere_benchmark.py`| L-BFGS-B (More-Thuente and Armijo) vs CMA-ES on 10D Sphere                                | `plots/lbfgsb/sphere_benchmark/`    |
| `initial_hessian.py` | Effect of `initial_hessian` and `persist_initial_hessian` on 10D Ellipsoid (cond 10⁶)     | `plots/lbfgsb/initial_hessian/`     |
| `rotation_study.py`  | Full-Hessian vs diagonal vs identity B₀ across 4 rotations × {n=10,m=10}, {n=50,m=5}; also produces landscape grids | `plots/lbfgsb/rotation_study/`      |

## `handoff/` — CMA-ES → L-BFGS-B handoff studies

The headline contribution of the thesis: hand the CMA-ES learned
covariance to L-BFGS-B as the initial Hessian B₀.

| Script                          | What it does                                                                                                 | Output                                            |
|---------------------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| `covariance_transformations.py` | 6 candidate `B₀` transforms (`I`, `C`, `C⁻¹`, `(σ²C)⁻¹`, normalized `C`, normalized `C⁻¹`) on rotated Ellipsoid | `plots/handoff/covariance_transformations/`       |
| `multimodal.py`                 | Rastrigin and Griewank: CMA-ES vs L-BFGS-B vs `C⁻¹` handoff vs identity handoff (25 seeds default)            | `plots/handoff/multimodal/`                       |
| `multimodal_rotated.py`         | Same comparison on `RotatedFunction(Rastrigin / Griewank)`; confirms rotation alone doesn't expose anisotropy | `plots/handoff/multimodal_rotated/`               |
| `rippled_ellipsoid.py`          | The decisive case: `RotatedFunction(RippledEllipsoid)` with high cond + low amplitude — `C⁻¹` dominates       | `plots/handoff/rippled_ellipsoid/`                |
| `timing_sweep_multimodal.py`    | Multimodal warmup sweep: 5 handoff timings × CMA-ES + L-BFGS-B baselines                                     | `plots/handoff/timing_sweep_multimodal/`          |
| `timing_sweep_rippled.py`       | Rippled-ellipsoid warmup sweep; demonstrates the non-monotone sweet spot                                     | `plots/handoff/timing_sweep_rippled/`             |
| `timing_sweep_families.py`      | Parameterized timing sweep across 4 named experimental families (`baseline`, `reproduce_old`, `low_amp`, `multimodal`) | `plots/report/timing_{family}/`                   |

Common flags (most scripts): `--num-seeds`, `--num-workers`,
`--total-budget`, `--cmaes-warmup-budget`, `--dimensions`,
`--output-dir`. Run any script with `--help` for its complete option list.

## `report/` — supervisor-report regeneration

| Script                  | What it does                                                                                              | Output                  |
|-------------------------|-----------------------------------------------------------------------------------------------------------|-------------------------|
| `regen_report_plots.py` | Re-renders the 5 supervisor-report figures from cached `traces.json` files under `plots/report/<panel>/`. No optimizer runs are needed. | `plots/report/<panel>/` |

## Adding a new experiment

1. Pick the right subdirectory (or create one if a new topic) and add a
   Python script.
2. Write outputs to `plots/<same-group>/<script-name>/`. Create the
   directory at runtime with `Path(...).mkdir(parents=True, exist_ok=True)`.
3. If the experiment uses the multi-seed framework, prefer
   `src.benchmarking.Benchmark` — you get persisted `traces.json` for
   free, which makes the result re-plottable without re-running.
4. For plotting, use the declarative API in `src.plotting`:
   - `plot_metrics(result)` for single-run diagnostics.
   - `plot_comparison(results)` for cross-algorithm comparison.
   - `plot_benchmark_convergence(...)` / `plot_benchmark_boxplot(...)`
     for multi-seed benchmark plots.
5. If you need a custom benchmark runner (something `SingleAlgorithm`
   doesn't cover), inherit from `BenchmarkAlgorithm` (or
   `HandoffAlgorithm` for two-phase runners). See `basic/custom_handoff.py`
   and `basic/custom_algorithm.py` for examples.
6. Document it in this README.
