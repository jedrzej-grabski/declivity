"""Empirical cumulative distribution of running times (ECDF).

An algorithm's ECDF curve answers: "of a shared set of target quality levels, what fraction has been reached by evaluation budget x?" — averaged across seeds. It complements the median/IQR convergence band in
:mod:`declivity.benchmarking.aggregation`: that view answers "how good is
the typical run at x", this one answers "how many of the targets we care
about are met by x", which stays comparable across algorithms with very
different convergence shapes (fast-then-stalling vs. slow-then-sharp).

Two grids drive the computation:

- a **threshold grid** (:func:`threshold_grid`) — log-spaced target fitness
  levels, built once from the pooled ``best_fitness`` range of every run you
  want to compare fairly (typically every algorithm on one problem, so
  they're judged against identical targets);
- a **budget grid** — log-spaced evaluation counts, the shared x-axis every
  algorithm's curve is discretised onto. Reuse
  :func:`declivity.benchmarking.aggregation.series_grid` for this; it isn't
  redefined here.

By default every function here works on the *gap to the known global
optimum* (``field value - global_minimum``). Pass ``global_minimum=`` the
problem's known ``f*`` (e.g. from ``BenchmarkFunction.global_minimum``);
pass ``normalize=False`` to fall back to raw values instead (no known
optimum, or reusing this machinery for a field that isn't a fitness gap).
"""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from declivity.benchmarking.aggregation import step_function_lookup
from declivity.benchmarking.run_trace import RunTrace

DEFAULT_THRESHOLD_FLOOR = 1e-8


def threshold_grid(
    traces: Iterable[RunTrace],
    n_thresholds: int = 50,
    floor: float = DEFAULT_THRESHOLD_FLOOR,
    field: str = "best_fitness",
    *,
    global_minimum: float = 0.0,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Log-spaced target levels spanning the pooled ``field`` range.

    Pool every trace you want compared on equal footing (typically every
    algorithm's runs on one problem) — the grid is shared across whatever
    is passed in, so callers control the fairness boundary. Falls back to
    ``[floor, floor * 10]`` when every value is at or below ``floor`` (e.g.
    a trivially easy problem), so degenerate inputs still produce a valid
    grid instead of a zero-width ``logspace`` call.

    When ``normalize`` (default), the range is taken over the *gap*
    ``field value - global_minimum`` rather than the raw values; a
    constant shift doesn't change where the max sits, so this is just
    ``max(values) - global_minimum``.
    """
    max_value = floor
    for trace in traces:
        values = trace.get_series(field)
        if values:
            observed = max(values) - global_minimum if normalize else max(values)
            max_value = max(max_value, observed)
    if max_value <= floor:
        max_value = floor * 10
    return np.logspace(np.log10(floor), np.log10(max_value), n_thresholds)


def run_ecdf(
    trace: RunTrace,
    thresholds: NDArray[np.float64],
    field: str = "best_fitness",
    x_field: str = "evaluations",
    *,
    global_minimum: float = 0.0,
    normalize: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One run's fraction-of-thresholds-hit step function.

    Returns ``(evaluations, frac)`` — ``frac[i]`` is the share of
    ``thresholds`` already ``>=`` the run's ``field`` value at
    ``evaluations[i]`` (gap-normalised when ``normalize``, i.e. compared
    against ``field value - global_minimum``), i.e. the fraction of
    targets reached so far. Monotone non-decreasing by construction since
    ``field`` (``best_fitness``) is itself monotone non-increasing and a
    constant shift preserves that, so no defensive running-max is needed
    before discretising.
    """
    xs = trace.get_series(x_field)
    ys = trace.get_series(field)
    if not xs or not ys:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    length = min(len(xs), len(ys))
    x_array = np.asarray(xs[:length], dtype=np.float64)
    y_array = np.asarray(ys[:length], dtype=np.float64)
    if normalize:
        y_array = y_array - global_minimum
    frac = np.mean(y_array[:, None] <= thresholds[None, :], axis=1)
    return x_array, frac


def aggregate_ecdf(
    traces: Iterable[RunTrace],
    thresholds: NDArray[np.float64],
    x_grid: NDArray[np.int64],
    field: str = "best_fitness",
    x_field: str = "evaluations",
    *,
    global_minimum: float = 0.0,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Mean ECDF curve across ``traces``, discretised onto ``x_grid``.

    One algorithm's runs in, one curve out. A run missing data contributes
    an all-zero curve (no targets reached) rather than being dropped, so a
    partially-failed benchmark still aggregates.
    """
    rows: list[NDArray[np.float64]] = []
    for trace in traces:
        x_array, frac = run_ecdf(
            trace,
            thresholds,
            field,
            x_field,
            global_minimum=global_minimum,
            normalize=normalize,
        )
        if x_array.size == 0:
            rows.append(np.zeros(x_grid.shape, dtype=np.float64))
            continue
        curve = step_function_lookup(x_array.tolist(), frac.tolist(), x_grid)
        # Grid cells before a run's first logged point come back as the
        # "no progress yet" +inf sentinel `step_function_lookup` uses for
        # fitness curves; for an ECDF that means zero targets reached yet,
        # not infinity.
        rows.append(np.where(np.isinf(curve), 0.0, curve))
    if not rows:
        return np.zeros(x_grid.shape, dtype=np.float64)
    return np.mean(rows, axis=0)


def ecdf_auc(x_grid: NDArray[np.int64], curve: NDArray[np.float64]) -> float:
    """Area under an ECDF curve, normalised by the grid's final x value."""
    return float(trapezoid(curve, x_grid) / x_grid[-1])
