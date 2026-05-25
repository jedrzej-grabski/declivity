"""Helpers to aggregate multiple RunTraces into a common evaluation grid.

Each trace is a step function: best_fitness[i] holds from evaluations[i]
up to evaluations[i+1]. To build per-evaluation summary statistics we
sample each trace on a shared grid and return one matrix per algorithm.
"""

import warnings
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from src.benchmarking.run_trace import RunTrace


def common_evaluation_grid(
    traces: Iterable[RunTrace],
    num_points: int = 200,
    log_spaced: bool = True,
    min_eval: int = 1,
) -> NDArray[np.int64]:
    """Build a shared evaluation grid spanning all traces.

    Log-spaced grids give roughly uniform resolution on a log-x convergence
    plot — useful since most progress happens early. Uses unique integers
    so we don't waste samples on duplicate ticks for short runs.
    """
    max_eval = max(trace.final_evaluations for trace in traces)
    if max_eval <= min_eval:
        return np.array([min_eval], dtype=np.int64)

    if log_spaced:
        grid = np.geomspace(min_eval, max_eval, num=num_points)
    else:
        grid = np.linspace(min_eval, max_eval, num=num_points)

    return np.unique(np.round(grid).astype(np.int64))


def step_function_lookup(
    evaluations: list[int],
    best_fitness: list[float],
    grid: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Sample a (evaluations, best_fitness) trace on a grid.

    Returns the best fitness reached at or before each grid point. Grid
    points before the first logged evaluation are filled with ``+inf``
    (no progress yet); points after the last logged evaluation are flat
    at the trace's final value.
    """
    eval_array = np.asarray(evaluations, dtype=np.int64)
    fitness_array = np.asarray(best_fitness, dtype=np.float64)

    indices = np.searchsorted(eval_array, grid, side="right") - 1
    out = np.empty_like(grid, dtype=np.float64)
    out[indices < 0] = np.inf
    valid = indices >= 0
    out[valid] = fitness_array[indices[valid]]
    return out


def stack_traces_on_grid(
    traces: Iterable[RunTrace],
    grid: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Stack multiple traces onto a shared grid. Returns shape (n_traces, n_grid)."""
    return np.array(
        [
            step_function_lookup(trace.evaluations, trace.best_fitness, grid)
            for trace in traces
        ],
        dtype=np.float64,
    )


def percentile_band(
    matrix: NDArray[np.float64],
    low_percentile: float = 25.0,
    high_percentile: float = 75.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (median, low, high) per column, ignoring +inf entries when possible.

    Early grid points may precede any algorithm's first logged evaluation —
    those columns are all-NaN, which numpy warns about. The warnings are
    benign (the corresponding grid point is just suppressed downstream), so
    we silence them locally.
    """
    cleaned = np.where(np.isinf(matrix), np.nan, matrix)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN axis encountered")
        median = np.nanmedian(cleaned, axis=0)
        low = np.nanpercentile(cleaned, low_percentile, axis=0)
        high = np.nanpercentile(cleaned, high_percentile, axis=0)
    return median, low, high
