"""Regression tests for :class:`InterleavedCMAESLocal`.

Covers the generalization from the old L-BFGS-B-only ``InterleavedCMAESLBFGSB``
(``local_algorithm=LBFGSB`` or ``BFGS``) and the ``HessianScaling.ADAPTIVE``
carryover threaded across bursts.
"""

import numpy as np

from declivity.algorithms.bfgs.config import BFGSConfig
from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.benchmarking import InterleavedCMAESLocal, Problem
from declivity.utils.benchmark_functions import RotatedEllipsoid, ShiftedFunction
from declivity.utils.initial_geometry import HessianScaling

DIMENSIONS = 6


def _problem() -> Problem:
    base = RotatedEllipsoid(DIMENSIONS, rotation="random", seed=0)
    func = ShiftedFunction.near_corner(base, fraction=0.9)
    return Problem.from_benchmark("test-interleaved", func)


def _cmaes_config(dimensions: int) -> CMAESConfig:
    return CMAESConfig(dimensions=dimensions)


def _run(local_algorithm: AlgorithmChoice, scaling: HessianScaling | str):
    problem = _problem()
    x0 = problem.starting_point(0)

    if local_algorithm == AlgorithmChoice.LBFGSB:
        local_config_factory = lambda d: LBFGSBConfig(  # noqa: E731
            dimensions=d, m=6, pgtol=1e-10, factr=0
        )
    else:
        local_config_factory = lambda d: BFGSConfig(dimensions=d)  # noqa: E731

    algorithm = InterleavedCMAESLocal(
        name=f"{local_algorithm}-{scaling}",
        color="#000000",
        cmaes_config_factory=_cmaes_config,
        local_config_factory=local_config_factory,
        local_algorithm=local_algorithm,
        cmaes_interval=5,
        total_budget=1500,
        transform="inverse",
        scaling=scaling,
        probe_factr=1e7,
        probe_pgtol=1e-8,
        probe_max_evals=60,
    )
    return algorithm.run_with_detail(problem, x0, seed=0)


def test_lbfgsb_local_runs_and_produces_bursts():
    detail = _run(AlgorithmChoice.LBFGSB, HessianScaling.NONE)
    assert detail.num_bursts > 0
    assert np.isfinite(detail.trace.final_fitness)


def test_bfgs_local_runs_and_produces_bursts():
    detail = _run(AlgorithmChoice.BFGS, HessianScaling.NONE)
    assert detail.num_bursts > 0
    assert np.isfinite(detail.trace.final_fitness)


def test_adaptive_scaling_runs_for_both_targets():
    for local_algorithm in (AlgorithmChoice.LBFGSB, AlgorithmChoice.BFGS):
        detail = _run(local_algorithm, HessianScaling.ADAPTIVE)
        assert detail.num_bursts > 0
        assert np.isfinite(detail.trace.final_fitness)


def test_invalid_local_algorithm_raises():
    problem = _problem()
    x0 = problem.starting_point(0)
    algorithm = InterleavedCMAESLocal(
        name="bad",
        color="#000000",
        cmaes_config_factory=_cmaes_config,
        local_config_factory=lambda d: LBFGSBConfig(dimensions=d),
        local_algorithm=AlgorithmChoice.POWELL,
    )
    try:
        algorithm.run_with_detail(problem, x0, seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unsupported local_algorithm")


if __name__ == "__main__":
    tests = [
        test_lbfgsb_local_runs_and_produces_bursts,
        test_bfgs_local_runs_and_produces_bursts,
        test_adaptive_scaling_runs_for_both_targets,
        test_invalid_local_algorithm_raises,
    ]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"{len(tests)} tests passed")
