# Backlog

- **exp1 conditioning: scaling sweep wastes budget on scale-invariant optimizers.**
  `HessianScaling` (`declivity/utils/initial_geometry.py`) applies a single
  isotropic scalar to `B_0`, magnitude-only by construction - Powell and
  Nelder-Mead read shape-only quantities (`principal_directions`/
  `principal_scales`/`axis_steps`, all normalized) and are mathematically
  invariant to it (see the docstring at `initial_geometry.py:208-212`). But
  `run_local_stage`/`build_contenders` in
  `experiments/conditioning/exp1_conditioners.py` builds one contender per
  `(optimizer, conditioner)` for every optimizer in `spec.optimizers` for
  *every* scaling in `spec.scalings`, with no scaling-aware filter. Sweeping
  a new scaling (e.g. the new adaptive one) re-runs Powell/Nelder-Mead for
  results identical (up to RNG) to `scaling="none"`, burning the full
  `local_budget_per_dim` for nothing unless you manually restrict
  `optimizers=[lbfgsb,bfgs]` per scaling run.
  Also: `run_local_stage` writes one `traces.parquet` per `(scaling, variant,
  dim)` via `Benchmark.run()` -> `save_traces_parquet`, which is a full
  overwrite, not a merge/append. So a scaling-restricted `optimizers=[...]`
  run for a *new* scaling doesn't "preserve" NM/Powell - it just never
  writes them into that scaling's file at all.
  Figure out: (a) whether `build_contenders` should skip scaling-invariant
  optimizers per scaling automatically (needs a per-optimizer
  "scale-invariant" flag, and confirming this holds for every current/future
  `HessianScaling` variant - all of them look isotropic today), and (b)
  whether local-stage persistence should support incremental/merge writes
  so a partial optimizer rerun doesn't require re-declaring every optimizer
  you want kept in that scaling's `traces.parquet`.
