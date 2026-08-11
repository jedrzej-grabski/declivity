# TODO

- `declivity/benchmarking/cmaes_path.py:record_cmaes_path` stops CMA-ES with
  `MaxIterations`, not `MaxEvaluations`. `MaxIterations` doesn't override
  `StoppingCondition.remaining_evaluations`, so
  `BaseOptimizer.evaluate_population` never trims the last generation — every
  slice runs a full population's worth of objective calls and only checks the
  cap between generations. That means any evaluation budget converted to
  `max_iterations` for this path (e.g. `experiments/conditioning/exp1_conditioners.py`'s
  `cmaes_evaluations_per_dim`) can overshoot by up to `population_size - 1`
  evaluations, unlike direct `MaxEvaluations` callers elsewhere in the repo
  (`algorithm_run.py`, `custom_algorithm.py`, etc.), which get an exact,
  objective-call-level cutoff. Fix: give `record_cmaes_path` (or its internal
  per-slice `CMAESOptimizer`) an evaluations-aware stopping condition —
  e.g. `MaxIterations(target) & MaxEvaluations(budget)` — so a slice near the
  end of the budget is trimmed instead of run to full-generation completion.
