"""Regression guard: an *identity* geometry seed reproduces the native baseline.

The uniform ``initial_geometry=`` seam added to Powell and L-BFGS-B must not
disturb the (scipy-identical) default paths. This checks the two contracts that
make that guarantee auditable:

1. ``initial_geometry=None`` leaves the native seam untouched — the run is the
   exact default (already validated bit-for-bit vs scipy in
   ``powell_vs_scipy.py``).

2. A neutral ``InitialGeometry.identity(n)`` seed reproduces the default run
   **exactly**, since both defaults are themselves isotropic:
   - **Powell**   — ``principal_directions()`` of an identity geometry is ``I``,
     i.e. the default coordinate direction set.
   - **L-BFGS-B** — an identity geometry is ``B_0 = I``, i.e. the default.

Every per-(function, dim, seed) trajectory is compared exactly: the running-best
list, the evaluation-count list, and the final solution vector. Exits non-zero on
any deviation.

Run::

    PYTHONPATH=. pdm run python experiments/cross_validation/geometry_seed_identity.py
"""

import argparse
import sys

import numpy as np

from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.algorithms.lbfgsb.lbfgsb_optimizer import LBFGSBOptimizer
from declivity.algorithms.powell.config import PowellConfig
from declivity.algorithms.powell.powell_optimizer import PowellOptimizer
from declivity.utils.benchmark_functions import Ellipsoid, Rosenbrock, Sphere
from declivity.utils.initial_geometry import InitialGeometry
from declivity.utils.stopping_conditions import MaxEvaluations

FUNCTIONS = [("Sphere", Sphere), ("Rosenbrock", Rosenbrock), ("Ellipsoid", Ellipsoid)]


def _trajectory(result) -> tuple[list[float], list[int], np.ndarray]:
    diag = result.diagnostic
    return (
        list(diag.best_fitness),
        list(diag.evaluations),
        np.asarray(result.best_solution, dtype=float),
    )


def _identical(a, b) -> bool:
    fa, ea, xa = a
    fb, eb, xb = b
    return (
        fa == fb
        and ea == eb
        and xa.shape == xb.shape
        and np.array_equal(xa, xb)
    )


def _run_powell(func, x0, budget, geometry):
    return PowellOptimizer(
        func, x0, PowellConfig(dimensions=len(x0)),
        stopping_condition=MaxEvaluations(budget), seed=0,
        initial_geometry=geometry,
    ).optimize()


def _run_lbfgsb(func, x0, budget, geometry):
    return LBFGSBOptimizer(
        func, x0, LBFGSBConfig(dimensions=len(x0)),
        stopping_condition=MaxEvaluations(budget),
        initial_geometry=geometry,
    ).optimize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=5000)
    args = parser.parse_args()

    exact_cases = [
        ("Powell", _run_powell),
        ("L-BFGS-B", _run_lbfgsb),
    ]

    failures = 0
    print("=" * 74)
    print("Identity-geometry seed vs native default (must be trajectory-exact)")
    print("=" * 74)

    for algo_name, runner in exact_cases:
        for fn_name, fn_cls in FUNCTIONS:
            for dim in args.dims:
                for seed in range(args.num_seeds):
                    rng = np.random.default_rng(seed)
                    x0 = rng.uniform(-80.0, 80.0, size=dim)
                    default = _trajectory(runner(fn_cls(dim), x0, args.budget, None))
                    seeded = _trajectory(
                        runner(fn_cls(dim), x0, args.budget, InitialGeometry.identity(dim))
                    )
                    ok = _identical(default, seeded)
                    failures += not ok
                    if not ok:
                        print(
                            f"  [FAIL] {algo_name:9} {fn_name:10} d={dim:<3} seed={seed}: "
                            f"identity-geometry trajectory != default"
                        )
        print(f"  {algo_name}: identity geometry == default across all "
              f"{len(FUNCTIONS)}x{len(args.dims)}x{args.num_seeds} cases "
              f"{'OK' if failures == 0 else 'with FAILURES'}")

    print("=" * 74)
    if failures:
        print(f"RESULT: {failures} FAILURE(S) — the geometry seam changed a default path.")
        sys.exit(1)
    print("RESULT: PASS — identity geometry reproduces the native baselines exactly.")


if __name__ == "__main__":
    main()
