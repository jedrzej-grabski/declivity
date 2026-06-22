"""Helpers to aggregate multiple runs onto a common x grid.

Each run is a step function: ``values[i]`` holds from ``x[i]`` up to
``x[i+1]``. To build per-x summary statistics we sample each run on a shared
grid and return one matrix (runs x grid).

The functions are **field-agnostic**: they read a run's per-step arrays via
``run.get_series(field)``, which both a live ``LogData`` and a persisted
``RunTrace`` implement. So the same median+band machinery that aggregates
``best_fitness`` across seeds also aggregates ``sigma`` or ``condition_number``
— whatever was retained. ``common_evaluation_grid`` / ``stack_traces_on_grid``
are kept as thin ``best_fitness``-on-``evaluations`` wrappers for existing
callers.
"""

import warnings
from typing import Iterable, Protocol

import numpy as np
from numpy.typing import NDArray


class _SeriesSource(Protocol):
    """Anything exposing named per-step arrays — a ``LogData`` or a ``RunTrace``."""

    def get_series(self, name: str) -> list[float] | None: ...


def series_grid(
    runs: Iterable[_SeriesSource],
    x_field: str = "evaluations",
    num_points: int = 200,
    log_spaced: bool = True,
    min_value: int = 1,
) -> NDArray[np.int64]:
    """Build a shared grid spanning all runs along ``x_field``.

    Log-spaced grids give roughly uniform resolution on a log-x convergence
    plot — useful since most progress happens early. Uses unique integers so
    we don't waste samples on duplicate ticks for short runs. Runs with no
    ``x_field`` data are ignored when computing the span.
    """
    max_value = min_value
    for run in runs:
        xs = run.get_series(x_field)
        if xs:
            max_value = max(max_value, int(round(float(xs[-1]))))

    if max_value <= min_value:
        return np.array([min_value], dtype=np.int64)

    if log_spaced:
        grid = np.geomspace(min_value, max_value, num=num_points)
    else:
        grid = np.linspace(min_value, max_value, num=num_points)
    return np.unique(np.round(grid).astype(np.int64))


def step_function_lookup(
    x_values: list[float],
    y_values: list[float],
    grid: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Sample a step-function run on a grid.

    Returns the value in effect at or before each grid point. Grid points
    before the first logged ``x`` are filled with ``+inf`` (no progress yet);
    points after the last are flat at the run's final value.
    """
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)

    indices = np.searchsorted(x_array, grid, side="right") - 1
    out = np.empty_like(grid, dtype=np.float64)
    out[indices < 0] = np.inf
    valid = indices >= 0
    out[valid] = y_array[indices[valid]]
    return out


def stack_runs_on_grid(
    runs: Iterable[_SeriesSource],
    grid: NDArray[np.int64],
    field: str = "best_fitness",
    x_field: str = "evaluations",
) -> NDArray[np.float64]:
    """Stack runs' ``field`` onto a shared grid. Returns shape (n_runs, n_grid).

    A run missing ``field`` or ``x_field`` contributes an all-NaN row, which
    :func:`percentile_band` ignores — so a field only some runs retained still
    aggregates over the runs that have it (rather than erroring).
    """
    rows: list[NDArray[np.float64]] = []
    nan_row = np.full(grid.shape, np.nan, dtype=np.float64)
    for run in runs:
        xs = run.get_series(x_field)
        ys = run.get_series(field)
        if not xs or not ys:
            rows.append(nan_row.copy())
            continue
        length = min(len(xs), len(ys))
        rows.append(step_function_lookup(xs[:length], ys[:length], grid))
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, grid.size))


def percentile_band(
    matrix: NDArray[np.float64],
    low_percentile: float = 25.0,
    high_percentile: float = 75.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (median, low, high) per column, ignoring +inf/NaN entries.

    Early grid points may precede any run's first logged step — those columns
    are all-NaN, which numpy warns about. The warnings are benign (the column
    is suppressed downstream), so we silence them locally.
    """
    cleaned = np.where(np.isinf(matrix), np.nan, matrix)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN axis encountered")
        median = np.nanmedian(cleaned, axis=0)
        low = np.nanpercentile(cleaned, low_percentile, axis=0)
        high = np.nanpercentile(cleaned, high_percentile, axis=0)
    return median, low, high


# ---------------------------------------------------------------------------
# Backward-compatible wrappers: best_fitness-on-evaluations over RunTraces.
# ---------------------------------------------------------------------------

def common_evaluation_grid(
    traces: Iterable[_SeriesSource],
    num_points: int = 200,
    log_spaced: bool = True,
    min_eval: int = 1,
) -> NDArray[np.int64]:
    """Shared evaluation grid spanning all traces (``series_grid`` on evals)."""
    return series_grid(
        list(traces), "evaluations", num_points, log_spaced, min_value=min_eval
    )


def stack_traces_on_grid(
    traces: Iterable[_SeriesSource],
    grid: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Stack traces' best_fitness onto a grid. Returns shape (n_traces, n_grid)."""
    return stack_runs_on_grid(list(traces), grid, "best_fitness", "evaluations")
