"""Check that the PoC loop with ``model_step=False`` *is* framework Nelder-Mead.

The whole PoC rests on being a clean ablation: the only difference between the
two arms must be the Hessian-completion block.  This asserts that the standalone
loop, run with the block disabled, reproduces
:class:`~declivity.algorithms.neldermead.NelderMeadOptimizer` (itself validated
bit-identical against SciPy) to machine precision.

    PYTHONPATH=. uv run python experiments/neldermead_curvature/validate.py
"""

from __future__ import annotations

import numpy as np

from declivity.algorithms.neldermead.config import NelderMeadConfig
from declivity.algorithms.neldermead.neldermead_optimizer import NelderMeadOptimizer
from declivity.utils.benchmark_functions import Ellipsoid, Rosenbrock, Sphere
from declivity.utils.constraint_handlers import BoxConstraintHandler, BoxStrategy
from declivity.utils.stopping_conditions import MaxEvaluations
from experiments.neldermead_curvature.hessian_completed import (
    HessianCompletedNelderMead,
)


def main() -> int:
    failures = 0
    for name, factory in (
        ("Sphere", Sphere),
        ("Ellipsoid", Ellipsoid),
        ("Rosenbrock", Rosenbrock),
    ):
        for dim in (5, 10):
            for seed in range(3):
                function = factory(dim)
                lower, upper = function.bounds
                rng = np.random.default_rng(seed)
                x0 = rng.uniform(lower, upper)
                budget = 400 * dim

                config = NelderMeadConfig(dimensions=dim, xatol=1e-8, fatol=1e-8)
                reference = NelderMeadOptimizer(
                    function,
                    x0,
                    config,
                    constraint_handler=BoxConstraintHandler(
                        BoxStrategy.CLAMP, lower, upper
                    ),
                    stopping_condition=MaxEvaluations(budget),
                    lower_bounds=lower,
                    upper_bounds=upper,
                    seed=seed,
                ).optimize()

                poc = HessianCompletedNelderMead(
                    function,
                    x0,
                    lower_bounds=lower,
                    upper_bounds=upper,
                    max_evaluations=budget,
                    model_step=False,
                    xatol=1e-8,
                    fatol=1e-8,
                ).optimize()

                delta_f = abs(poc.best_fitness - float(reference.best_fitness))
                scale = max(abs(float(reference.best_fitness)), 1e-30)
                delta_x = float(
                    np.max(np.abs(poc.best_solution - reference.best_solution))
                )
                delta_evals = abs(poc.evaluations - reference.evaluations)
                ok = delta_f / scale < 1e-12 and delta_x < 1e-10 and delta_evals == 0
                failures += not ok
                print(
                    f"{'OK  ' if ok else 'FAIL'} {name:<10} d={dim:<3} seed={seed}  "
                    f"rel df={delta_f / scale:.2e}  max dx={delta_x:.2e}  "
                    f"devals={delta_evals}"
                )

    print("\nall matched" if failures == 0 else f"\n{failures} mismatches")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
