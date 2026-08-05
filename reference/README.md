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

The R source for DES lives at `thesis-experiments/des_comparison/DES.R`.
Its literal Python port — used as the cross-validation oracle for the
framework-native `DESOptimizer` — lives at
`src/algorithms/des/des_reference.py`.

The R source for MF-CMA-ES lives at
`../../nm-cma-es-vectorized.R` (a bundled copy is in `mf_cmaes.r`).
Its literal Python port lives at
`src/algorithms/mfcmaes/mfcmaes_reference.py`.

## `outputs/`

Historical CSVs from an earlier ad-hoc comparison of Python DES against
hand-run R outputs. Superseded by the harness described below; kept for
provenance only.

| File                                  | Format                                            |
|---------------------------------------|---------------------------------------------------|
| `python_convergence_f<id>_d<dim>.csv` | `[max_iter, runs]` matrix of best-so-far fitness |
| `python_summary_f<id>_d<dim>.csv`     | `run, final_fitness, evaluations, runtime`        |

## Cross-validation harnesses

- **DES:** `experiments/cross_validation/des_vs_reference.py` runs both
  the framework-native `DESOptimizer` and the literal port
  `des_reference.des_reference` over a shared seed set, writes
  `plots/cross_validation/des_vs_reference/{convergence,distribution,state}_*.png`,
  `summary.csv`, and `parity_report.txt` (Wilcoxon + KS tests on final
  fitness). Defaults: CEC2017 F10, `d=10`, 25 seeds.
- **MF-CMA-ES:** `experiments/cross_validation/mfcmaes_vs_reference.py`
  does the analogous job for MF-CMA-ES against
  `mfcmaes_reference.nm_cma_es_vectorized`.
- **CMA-ES:** `experiments/cross_validation/cmaes_vs_reference.py` does
  the analogous job for CMA-ES against `cmaes_reference.CMA`.

To run:

```bash
PYTHONPATH=. uv run python -m experiments.cross_validation.des_vs_reference
PYTHONPATH=. uv run python -m experiments.cross_validation.mfcmaes_vs_reference
PYTHONPATH=. uv run python -m experiments.cross_validation.cmaes_vs_reference
```
