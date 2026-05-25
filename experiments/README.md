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

## `basic/` — tutorial demos and sanity checks

| Script                       | What it does                                                    | Output                                  |
|------------------------------|-----------------------------------------------------------------|-----------------------------------------|
| `simple_optimization.py`     | DES on a 10D Sphere; smallest end-to-end demo                  | `plots/basic/simple_optimization/`      |
| `covariance_adaptation.py`   | CMA-ES on Sphere / Ellipsoid; visualizes empirical covariance evolution (eigenvalues, condition, 2D ellipse snapshots) | `plots/basic/covariance_adaptation/`    |

## `cross_validation/` — checks against external references

| Script               | What it does                                                                          | Output                                              |
|----------------------|---------------------------------------------------------------------------------------|-----------------------------------------------------|
| `des_vs_r.py`        | Runs DES on CEC2017 F10 (10D, 10 seeds, fixed `x0=50·1`); writes convergence + summary CSVs that can be diffed against output from `reference/cmaes.r` / `reference/mf_cmaes.r` | `reference/outputs/python_*_f10_d10.csv`            |

## `lbfgsb/` — L-BFGS-B feature studies

| Script               | What it does                                                                              | Output                              |
|----------------------|-------------------------------------------------------------------------------------------|-------------------------------------|
| `sphere_benchmark.py`| L-BFGS-B (More-Thuente and Armijo) vs CMA-ES on 10D Sphere                                | `plots/lbfgsb/sphere_benchmark/`    |
| `initial_hessian.py` | Effect of `initial_hessian` and `persist_initial_hessian` on 10D Ellipsoid (cond 10⁶)     | `plots/lbfgsb/initial_hessian/`     |
| `rotation_study.py`  | Full-Hessian vs diagonal vs identity B₀ across 4 rotations × {n=10,m=10}, {n=50,m=5}     | `plots/lbfgsb/rotation_study/`      |

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
| `_legacy_cmaes_to_lbfgsb.py`    | Earlier handoff prototype; reaches into private CMA-ES attributes. Kept for archival; use the others instead | `plots/handoff/_legacy/`                          |

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
4. Document it in this README.
