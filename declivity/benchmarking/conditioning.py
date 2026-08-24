"""Seed local optimizers from persisted CMA-ES state, and compose
switch-interval hybrids offline.

Complements :mod:`declivity.benchmarking.cmaes_path`: a recorded path supplies
snapshots, and everything here turns a snapshot (or any curvature source) into
per-optimizer constructor kwargs, runs conditioned locals inside a
:class:`~declivity.benchmarking.Benchmark`, and stitches CMA-ES + probe traces
into the composed convergence curve of an interleaved hybrid: without ever
re-running the CMA-ES.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.benchmarking.algorithm_run import BenchmarkAlgorithm
from declivity.benchmarking.cmaes_path import CMAESSnapshot
from declivity.benchmarking.problem import Problem
from declivity.benchmarking.run_trace import RunTrace, capture_scalar_series
from declivity.core.algorithm_factory import AlgorithmFactory
from declivity.core.base_optimizer import BaseOptimizer, OptimizationResult
from declivity.core.config_base import BaseConfig
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.gradient_strategies import GradientStrategy
from declivity.utils.initial_geometry import (
    HandoffTransform,
    HessianScaling,
    InitialGeometry,
)
from declivity.utils.line_search import LineSearchStrategy
from declivity.utils.stopping_conditions import StoppingCondition

LOCAL_ALGORITHMS = (
    AlgorithmChoice.LBFGSB,
    AlgorithmChoice.BFGS,
    AlgorithmChoice.POWELL,
    AlgorithmChoice.NELDERMEAD,
    AlgorithmChoice.NELDERMEAD_HC,
)

_SIMPLEX_ALGORITHMS = (AlgorithmChoice.NELDERMEAD, AlgorithmChoice.NELDERMEAD_HC)
"""The optimizers whose initial population is a simplex, so they accept a
``simplex_base_size``."""

_POWELL_RATIO_FLOOR = 1e-6
"""Relative floor on scaled Powell direction lengths: keeps a collapsed
covariance axis from producing a numerically rank-deficient direction set."""


def snapshot_geometry(
    snapshot: CMAESSnapshot,
    transform: HandoffTransform | str = HandoffTransform.INVERSE,
    scaling: HessianScaling | str = HessianScaling.NONE,
) -> InitialGeometry:
    """The :class:`InitialGeometry` a snapshot's covariance defines.

    ``transform`` selects the curvature *shape*, ``scaling`` a separate
    magnitude factor applied on top (see :class:`HessianScaling`).
    """
    return InitialGeometry.from_covariance(
        snapshot.eigenvectors,
        snapshot.eigenvalues_sqrt,
        snapshot.sigma,
        transform,
        scaling=scaling,
    )


def local_seeding_kwargs(
    algorithm: AlgorithmChoice,
    geometry: InitialGeometry | None,
    scale_powell_directions: bool = True,
    simplex_base_size: float | None = None,
) -> dict[str, Any]:
    """Constructor kwargs that seed ``algorithm`` with ``geometry``.

    Powell receives the eigenvectors scaled by the covariance's per-axis
    standard deviations (``sqrt`` of the eigenvalues), normalised so the
    longest direction has unit length; ``scale_powell_directions=False`` keeps
    the unit-eigenvector default of the ``initial_geometry=`` seam.
    """
    if algorithm not in LOCAL_ALGORITHMS:
        names = ", ".join(str(choice) for choice in LOCAL_ALGORITHMS)
        raise ValueError(f"algorithm must be one of {names}; got {algorithm!r}.")
    if geometry is None:
        return {}

    if algorithm is AlgorithmChoice.POWELL and scale_powell_directions:
        directions = geometry.axis_steps(
            base_size=1.0, normalize=True, ratio_floor=_POWELL_RATIO_FLOOR
        )
        return {"initial_directions": directions.T}

    kwargs: dict[str, Any] = {"initial_geometry": geometry}
    if algorithm in _SIMPLEX_ALGORITHMS and simplex_base_size is not None:
        kwargs["simplex_base_size"] = simplex_base_size
    return kwargs


def run_conditioned_local(
    algorithm: AlgorithmChoice,
    problem: Problem,
    x0: NDArray[np.float64],
    config: BaseConfig,
    geometry: InitialGeometry | None,
    constraint_handler: ConstraintHandler,
    stopping_condition: StoppingCondition | None = None,
    seed: int | None = None,
    line_search: LineSearchStrategy | None = None,
    gradient_strategy: GradientStrategy | None = None,
    scale_powell_directions: bool = True,
    simplex_base_size: float | None = None,
) -> tuple[OptimizationResult, BaseOptimizer]:
    """One conditioned local run; returns the result and the optimizer, so a
    caller can also read the final learned state (``final_inverse_hessian`` /
    ``final_directions`` / ``final_simplex`` / ``final_corrections``)."""
    kwargs: dict[str, Any] = local_seeding_kwargs(
        algorithm,
        geometry,
        scale_powell_directions=scale_powell_directions,
        simplex_base_size=simplex_base_size,
    )
    if algorithm in (AlgorithmChoice.LBFGSB, AlgorithmChoice.BFGS):
        if problem.gradient is not None:
            kwargs["gradient_fn"] = problem.gradient
        if line_search is not None:
            kwargs["line_search"] = line_search
        if gradient_strategy is not None:
            kwargs["gradient_strategy"] = gradient_strategy

    optimizer = AlgorithmFactory.create_optimizer(
        algorithm,
        problem.function,
        x0,
        config,
        constraint_handler=constraint_handler,
        stopping_condition=stopping_condition,
        lower_bounds=problem.lower_bound,
        upper_bounds=problem.upper_bound,
        seed=seed,
        **kwargs,
    )
    return optimizer.optimize(), optimizer


@dataclass
class ConditionedLocalAlgorithm(BenchmarkAlgorithm):
    """A local optimizer whose initial geometry comes from a per-run provider.

    The provider maps ``(problem, seed)`` to the conditioner for that run -
    typically a persisted CMA-ES snapshot's covariance, a numerical Hessian,
    or ``InitialGeometry.identity`` for the control.  The optimizer itself
    starts from the benchmark's shared ``x0``, so contenders differ *only* in
    their conditioner.
    """

    name: str
    color: str
    algorithm: AlgorithmChoice
    config_factory: Callable[[int], BaseConfig]
    geometry_provider: Callable[[Problem, int], InitialGeometry | None]

    stopping_condition: StoppingCondition | None = None
    line_search: LineSearchStrategy | None = None
    gradient_strategy: GradientStrategy | None = None
    scale_powell_directions: bool = True
    simplex_base_size: float | None = None
    constraint_handler: ConstraintHandler | None = None

    record: Callable[[Problem, int, OptimizationResult, BaseOptimizer], None] | None = (
        None
    )
    """Optional per-run hook, called after ``optimize()``: e.g. to persist
    the run's final learned state and configuration next to the snapshots."""

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        result, optimizer = run_conditioned_local(
            self.algorithm,
            problem,
            x0,
            self.config_factory(problem.dimensions),
            self.geometry_provider(problem, seed),
            constraint_handler=self.resolve_constraint_handler(problem),
            stopping_condition=self.stopping_condition,
            seed=seed,
            line_search=self.line_search,
            gradient_strategy=self.gradient_strategy,
            scale_powell_directions=self.scale_powell_directions,
            simplex_base_size=self.simplex_base_size,
        )
        if self.record is not None:
            self.record(problem, seed, result, optimizer)
        return self.trace_from_result(problem, seed, result)


def probe_trace(
    problem: Problem,
    seed: int,
    result: OptimizationResult,
    algorithm_name: str,
) -> RunTrace:
    """Package a probe's :class:`OptimizationResult` as a bare :class:`RunTrace`."""
    return RunTrace(
        algorithm=algorithm_name,
        problem=problem.name,
        seed=seed,
        evaluations=[int(e) for e in result.diagnostic.evaluations],
        best_fitness=[float(f) for f in result.diagnostic.best_fitness],
        final_evaluations=result.evaluations,
        final_fitness=float(result.best_fitness),
        series=capture_scalar_series(result.diagnostic, retain=()),
    )


def retag_trace(trace: RunTrace, algorithm_name: str) -> RunTrace:
    """A copy of ``trace`` under a different algorithm label."""
    return replace(trace, algorithm=algorithm_name)


def compose_switch_trace(
    cmaes_trace: RunTrace,
    probes: list[tuple[int, RunTrace]],
    algorithm_name: str,
    first_switch_iteration: int | None = None,
) -> RunTrace:
    """Compose an interleaved-hybrid convergence curve offline.

    ``probes`` holds ``(switch_evaluations, probe_trace)`` pairs: each probe
    was launched from the CMA-ES state reached at ``switch_evaluations``
    cumulative CMA-ES evaluations.  The composed timeline inserts every
    probe's evaluations at its switch point, shifting the CMA-ES tail right,
    and takes the running minimum across both: the CMA-ES run itself is
    untouched (probes are side-branches, exactly the offline equivalent of an
    interleaved run without feedback).
    """
    events = sorted(probes, key=lambda pair: pair[0])
    cmaes_evals = cmaes_trace.evaluations
    cmaes_best = cmaes_trace.best_fitness

    composed_evals: list[int] = []
    composed_best: list[float] = []
    running_best = float("inf")
    offset = 0
    index = 0

    for switch_evaluations, probe in events:
        while index < len(cmaes_evals) and cmaes_evals[index] <= switch_evaluations:
            running_best = min(running_best, cmaes_best[index])
            composed_evals.append(cmaes_evals[index] + offset)
            composed_best.append(running_best)
            index += 1
        for probe_eval, probe_fitness in zip(probe.evaluations, probe.best_fitness):
            running_best = min(running_best, probe_fitness)
            composed_evals.append(switch_evaluations + offset + int(probe_eval))
            composed_best.append(running_best)
        offset += probe.final_evaluations

    while index < len(cmaes_evals):
        running_best = min(running_best, cmaes_best[index])
        composed_evals.append(cmaes_evals[index] + offset)
        composed_best.append(running_best)
        index += 1

    return RunTrace(
        algorithm=algorithm_name,
        problem=cmaes_trace.problem,
        seed=cmaes_trace.seed,
        evaluations=composed_evals,
        best_fitness=composed_best,
        final_evaluations=cmaes_trace.final_evaluations + offset,
        final_fitness=running_best,
        handoff_eval=events[0][0] if events else None,
        handoff_iter=first_switch_iteration,
    )
