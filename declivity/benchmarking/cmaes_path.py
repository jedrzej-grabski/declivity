"""Record one CMA-ES run together with periodic state snapshots.

The expensive warm-up is executed once per (problem, seed) and persisted;
conditioner studies and switch-interval hybrids then read the snapshots from
disk instead of re-running CMA-ES.  The run advances in slices through the
``CMAESState`` resume machinery with a shared RNG, which reproduces a
continuous run bit-for-bit, so the recorded path is exactly what a plain
``optimize()`` call would have produced.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.cmaes.cmaes_optimizer import CMAESOptimizer, CMAESState
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.benchmarking.persistence import (
    load_arrays_parquet,
    load_traces_parquet,
    save_arrays_parquet,
    save_traces_parquet,
)
from declivity.benchmarking.problem import Problem
from declivity.benchmarking.run_trace import RunTrace
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.stopping_conditions import MaxEvaluations, MaxIterations

_SNAPSHOT_ARRAY_FIELDS = (
    "sigma",
    "mean",
    "covariance",
    "evolution_path_c",
    "evolution_path_sigma",
    "eigenvectors",
    "eigenvalues_sqrt",
    "funhist_values",
    "x_best",
    "f_best",
)


@dataclass(frozen=True)
class CMAESSnapshot:
    """Full CMA-ES state at one recorded iteration, plus the incumbent."""

    iteration: int
    evaluations: int
    sigma: float
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    evolution_path_c: NDArray[np.float64]
    evolution_path_sigma: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    eigenvalues_sqrt: NDArray[np.float64]
    funhist_values: NDArray[np.float64]
    x_best: NDArray[np.float64]
    f_best: float

    def to_state(self) -> CMAESState:
        """Rebuild a resumable :class:`CMAESState` from this snapshot."""
        return CMAESState(
            mean=self.mean.copy(),
            sigma=float(self.sigma),
            covariance=self.covariance.copy(),
            evolution_path_c=self.evolution_path_c.copy(),
            evolution_path_sigma=self.evolution_path_sigma.copy(),
            generation=int(self.iteration),
            funhist_values=self.funhist_values.copy(),
            eigenvectors=self.eigenvectors.copy(),
            eigenvalues_sqrt=self.eigenvalues_sqrt.copy(),
        )


@dataclass
class CMAESPath:
    """One recorded CMA-ES run: its convergence trace + state snapshots."""

    trace: RunTrace
    snapshots: list[CMAESSnapshot]
    meta: dict = field(default_factory=dict)

    def snapshot_at(self, iteration: int) -> CMAESSnapshot | None:
        """The snapshot recorded exactly at ``iteration``, or ``None``."""
        for snapshot in self.snapshots:
            if snapshot.iteration == iteration:
                return snapshot
        return None

    def snapshot_at_or_before(self, iteration: int) -> CMAESSnapshot | None:
        """The latest snapshot at or before ``iteration``.

        Falls back to the final recorded state when the run terminated before
        ``iteration``: CMA-ES has stopped evolving by then, so its state at the
        requested iteration *is* the terminal one.  ``None`` only when nothing
        was recorded at or before it.
        """
        candidates = [s for s in self.snapshots if s.iteration <= iteration]
        return max(candidates, key=lambda s: s.iteration) if candidates else None


def record_cmaes_path(
    problem: Problem,
    x0: NDArray[np.float64],
    seed: int,
    config_factory: Callable[[int], CMAESConfig],
    snapshot_interval: int,
    max_evaluations: int,
    constraint_handler: ConstraintHandler | None = None,
    algorithm_name: str = "CMA-ES",
) -> CMAESPath:
    """Run CMA-ES on ``problem`` from ``x0``, snapshotting every
    ``snapshot_interval`` generations, stopping as soon as ``max_evaluations``
    objective calls is reached or exceeded.

    CMA-ES's generation is atomic (a full population is needed to rank and
    update the distribution), so the budget check is only between
    generations, not within one -- the final generation can overshoot by up
    to a population's worth of evaluations, unlike single-solution
    ``MaxEvaluations`` callers elsewhere in this codebase that trim mid-batch.

    Internal convergence (``tolfun`` / ``tolx`` / conditioning) ends the path
    early; the final state is still recorded (at its actual iteration) so a
    consumer can distinguish requested from reached snapshots via
    ``snapshot.iteration``.
    """
    if snapshot_interval <= 0 or max_evaluations <= 0:
        raise ValueError("snapshot_interval and max_evaluations must be positive.")

    rng = np.random.default_rng(seed)
    handler = constraint_handler or problem.resolved_constraint_handler()

    state: CMAESState | None = None
    cumulative_evaluations = 0
    best_fitness = float("inf")
    best_solution = np.asarray(x0, dtype=float).copy()

    trace_evaluations: list[int] = []
    trace_best: list[float] = []
    sigma_series: list[float] = []
    iteration_series: list[float] = []
    snapshots: list[CMAESSnapshot] = []
    message = ""

    target = 0
    while cumulative_evaluations < max_evaluations:
        target += snapshot_interval
        config = config_factory(problem.dimensions)
        remaining = max_evaluations - cumulative_evaluations
        optimizer = CMAESOptimizer(
            problem.function,
            x0,
            config,
            constraint_handler=handler,
            stopping_condition=MaxIterations(target) | MaxEvaluations(remaining),
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=rng,
            initial_state=state,
        )
        result = optimizer.optimize()
        message = result.message

        diagnostic = result.diagnostic
        for evaluation, fitness, sigma_value, generation in zip(
            diagnostic.evaluations,
            diagnostic.best_fitness,
            diagnostic.sigma,
            diagnostic.iteration,
        ):
            best_fitness = min(best_fitness, float(fitness))
            trace_evaluations.append(cumulative_evaluations + int(evaluation))
            trace_best.append(best_fitness)
            sigma_series.append(float(sigma_value))
            iteration_series.append(float(generation))
        if float(result.best_fitness) <= best_fitness:
            best_fitness = float(result.best_fitness)
            best_solution = np.asarray(result.best_solution, dtype=float).copy()

        cumulative_evaluations += result.evaluations
        state = optimizer.get_state()

        # get_state() caches (B, D) after every generation, so both are set.
        assert state.eigenvectors is not None and state.eigenvalues_sqrt is not None
        snapshots.append(
            CMAESSnapshot(
                iteration=int(state.generation),
                evaluations=cumulative_evaluations,
                sigma=float(state.sigma),
                mean=state.mean.copy(),
                covariance=state.covariance.copy(),
                evolution_path_c=state.evolution_path_c.copy(),
                evolution_path_sigma=state.evolution_path_sigma.copy(),
                eigenvectors=state.eigenvectors.copy(),
                eigenvalues_sqrt=state.eigenvalues_sqrt.copy(),
                funhist_values=state.funhist_values.copy(),
                x_best=best_solution.copy(),
                f_best=best_fitness,
            )
        )

        if state.generation < target:
            break

    trace = RunTrace(
        algorithm=algorithm_name,
        problem=problem.name,
        seed=seed,
        evaluations=trace_evaluations,
        best_fitness=trace_best,
        final_evaluations=cumulative_evaluations,
        final_fitness=best_fitness,
        series={"sigma": sigma_series, "iteration": iteration_series},
    )
    meta = {
        "dimensions": problem.dimensions,
        "seed": seed,
        "snapshot_interval": snapshot_interval,
        "max_evaluations": max_evaluations,
        "population_size": config_factory(problem.dimensions).population_size,
        "message": message,
    }
    return CMAESPath(trace=trace, snapshots=snapshots, meta=meta)


def save_cmaes_path(directory: str | Path, path_record: CMAESPath) -> Path:
    """Persist a :class:`CMAESPath` as ``trace.parquet`` + ``snapshots.parquet``
    + ``meta.json`` under ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    trace = path_record.trace
    save_traces_parquet(
        {(trace.problem, trace.algorithm): [trace]}, directory / "trace.parquet"
    )

    snapshots = path_record.snapshots
    columns: dict[str, list] = {
        "iteration": [s.iteration for s in snapshots],
        "evaluations": [s.evaluations for s in snapshots],
    }
    for name in _SNAPSHOT_ARRAY_FIELDS:
        columns[name] = [getattr(s, name) for s in snapshots]
    save_arrays_parquet(directory / "snapshots.parquet", columns)

    (directory / "meta.json").write_text(json.dumps(path_record.meta, indent=2))
    return directory


def load_cmaes_path(directory: str | Path) -> CMAESPath:
    """Load a :class:`CMAESPath` previously written by :func:`save_cmaes_path`."""
    directory = Path(directory)
    traces = load_traces_parquet(directory / "trace.parquet")
    (trace_list,) = traces.values()
    (trace,) = trace_list

    arrays = load_arrays_parquet(directory / "snapshots.parquet")
    count = len(arrays["iteration"])
    snapshots: list[CMAESSnapshot] = [
        CMAESSnapshot(
            iteration=int(arrays["iteration"][i]),
            evaluations=int(arrays["evaluations"][i]),
            sigma=float(arrays["sigma"][i]),
            mean=arrays["mean"][i],
            covariance=arrays["covariance"][i],
            evolution_path_c=arrays["evolution_path_c"][i],
            evolution_path_sigma=arrays["evolution_path_sigma"][i],
            eigenvectors=arrays["eigenvectors"][i],
            eigenvalues_sqrt=arrays["eigenvalues_sqrt"][i],
            funhist_values=arrays["funhist_values"][i],
            x_best=arrays["x_best"][i],
            f_best=float(arrays["f_best"][i]),
        )
        for i in range(count)
    ]

    meta = json.loads((directory / "meta.json").read_text())
    return CMAESPath(trace=trace, snapshots=snapshots, meta=meta)
