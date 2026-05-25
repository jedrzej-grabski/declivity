# Reference implementations and outputs

External-language reference implementations of algorithms ported in
this thesis, kept here for cross-validation. The Python ports in
`src/algorithms/` are the canonical implementations — these are for
checking against.

## Files

| File              | Language | Notes                                                                          |
|-------------------|----------|--------------------------------------------------------------------------------|
| `cmaes.r`         | R        | Reference CMA-ES used as the cross-validation baseline                        |
| `mf_cmaes.r`      | R        | Reference matrix-free CMA-ES (Arabas), with PPMF step-size adaptation         |

## `outputs/`

CSVs written by `experiments/cross_validation/des_vs_r.py` for direct
comparison against R-side outputs:

| File                                  | Origin                                          | Format                                            |
|---------------------------------------|-------------------------------------------------|---------------------------------------------------|
| `python_convergence_f<id>_d<dim>.csv` | Python DES, one column per seed (`run_1`, `run_2`, …) | `[max_iter, runs]` matrix of best-so-far fitness |
| `python_summary_f<id>_d<dim>.csv`     | Python DES, one row per seed                    | `run, final_fitness, evaluations, runtime` columns |

The current `python_*_f10_d10.csv` files were generated against CEC2017
F10 with `dimensions=10`, `seed=42+run`, and `np.full(10, 50.0)` as the
fixed initial point — matching the setup in the R reference for direct
diff.

To regenerate:

```bash
pdm run run-r
# or
PYTHONPATH=. pdm run python -m experiments.cross_validation.des_vs_r
```

## How comparison is performed

There is no automated comparison harness today. The CSV outputs are
diffed by eye against R outputs (run `cmaes.r` / `mf_cmaes.r` in R with
the same `function_id`, `dimensions`, and `seed` to get matching
columns). Aligning the seeds and `x0` is the responsibility of whoever
runs the comparison.
