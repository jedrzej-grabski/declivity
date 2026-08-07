"""Empirical cumulative distribution of running times (ECDF).

An algorithm's ECDF curve gives the fraction of a shared set of target
quality levels reached by evaluation budget x, averaged across seeds.  It
complements the median/IQR convergence band in
:mod:`declivity.benchmarking.aggregation` and stays comparable across
algorithms with different convergence shapes.

Two grids drive the computation:

- a threshold grid (:func:`threshold_grid`): log-spaced target fitness
  levels, built once from the pooled ``best_fitness`` range of every run
  being compared.  Pass ``ceiling=`` to fix the top of the range, which is
  needed for curves to be comparable across separate figures;
- a budget grid: log-spaced evaluation counts, the shared x-axis every
  algorithm's curve is discretised onto.  Use
  :func:`declivity.benchmarking.aggregation.series_grid`.

By default every function here works on the gap to the known global optimum
(``field value - global_minimum``).  ``BenchmarkFunction.global_minimum`` is
the pair ``(x*, f*)``, so pass its second element.
:func:`declivity.plotting.plot_benchmark_ecdf` reads it off the ``Problem``.
Pass ``gap_to_optimum=False`` to use raw values.

Runs that never reach a target stay in the denominator, so a curve plateaus
below 1.0 rather than being inflated by dropping them.
"""

import warnings
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from declivity.benchmarking.aggregation import _SeriesSource, step_function_lookup

DEFAULT_THRESHOLD_FLOOR = 1e-8


def threshold_grid(
    traces: Iterable[_SeriesSource],
    n_thresholds: int = 50,
    floor: float = DEFAULT_THRESHOLD_FLOOR,
    field: str = "best_fitness",
    *,
    global_minimum: float = 0.0,
    gap_to_optimum: bool = True,
    ceiling: float | None = None,
) -> NDArray[np.float64]:
    """Log-spaced target levels spanning the pooled ``field`` range.

    The grid is shared across whatever is passed in, so pool every trace to
    be compared on equal footing.  Falls back to ``[floor, floor * 10]``
    when every value is at or below ``floor``, so degenerate inputs still
    give a valid grid instead of a zero-width ``logspace`` call.

    With ``gap_to_optimum`` (default) the range is taken over
    ``field value - global_minimum``, which reduces to
    ``max(values) - global_minimum``.

    Non-finite values are ignored when sizing the range.  DES logs its first
    iteration before the incumbent is set, so every DES trace starts with
    ``+inf``, which would otherwise make the whole grid
    ``[nan, inf, inf, ...]`` and pin every curve to ~1.0.

    ``ceiling`` fixes the top of the range explicitly, bypassing the data.
    """
    if ceiling is not None:
        if not np.isfinite(ceiling) or ceiling <= floor:
            raise ValueError(
                f"ceiling must be finite and greater than floor={floor}, got {ceiling}"
            )
        return np.logspace(np.log10(floor), np.log10(ceiling), n_thresholds)

    max_value = floor
    for trace in traces:
        values = trace.get_series(field)
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        observed = float(finite.max())
        if gap_to_optimum:
            observed -= global_minimum
        max_value = max(max_value, observed)
    if max_value <= floor:
        max_value = floor * 10
    return np.logspace(np.log10(floor), np.log10(max_value), n_thresholds)


def run_ecdf(
    trace: _SeriesSource,
    thresholds: NDArray[np.float64],
    field: str = "best_fitness",
    x_field: str = "evaluations",
    *,
    global_minimum: float = 0.0,
    gap_to_optimum: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One run's fraction-of-thresholds-hit step function.

    Returns ``(evaluations, frac)`` — ``frac[i]`` is the share of
    ``thresholds`` already ``>=`` the run's ``field`` value at
    ``evaluations[i]`` (gap-normalised when ``gap_to_optimum``, i.e.
    compared against ``field value - global_minimum``), i.e. the fraction
    of targets reached so far. Monotone non-decreasing by construction
    since ``field`` (``best_fitness``) is itself monotone non-increasing
    and a constant shift preserves that, so no defensive running-max is
    needed before discretising.

    A non-finite ``field`` value counts as reaching no target, which is
    what the ``+inf`` DES logs before its first incumbent means.
    """
    xs = trace.get_series(x_field)
    ys = trace.get_series(field)
    if not xs or not ys:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    length = min(len(xs), len(ys))
    x_array = np.asarray(xs[:length], dtype=np.float64)
    y_array = np.asarray(ys[:length], dtype=np.float64)
    if gap_to_optimum:
        y_array = y_array - global_minimum
    reached = np.isfinite(y_array)[:, None] & (y_array[:, None] <= thresholds[None, :])
    frac = np.mean(reached, axis=1)
    return x_array, frac


def aggregate_ecdf(
    traces: Iterable[_SeriesSource],
    thresholds: NDArray[np.float64],
    x_grid: NDArray[np.int64],
    field: str = "best_fitness",
    x_field: str = "evaluations",
    *,
    global_minimum: float = 0.0,
    gap_to_optimum: bool = True,
) -> NDArray[np.float64]:
    """Mean ECDF curve across ``traces``, discretised onto ``x_grid``.

    One algorithm's runs in, one curve out.  A run missing data contributes
    an all-zero curve rather than being dropped, so a partially-failed
    benchmark still aggregates; it does drag the mean down, hence the
    warning.  ``stack_runs_on_grid`` differs here: it appends an all-NaN row
    the percentile band ignores, whereas a run that produced nothing did
    reach no target.
    """
    rows: list[NDArray[np.float64]] = []
    empty = 0
    for trace in traces:
        x_array, frac = run_ecdf(
            trace,
            thresholds,
            field,
            x_field,
            global_minimum=global_minimum,
            gap_to_optimum=gap_to_optimum,
        )
        if x_array.size == 0:
            empty += 1
            rows.append(np.zeros(x_grid.shape, dtype=np.float64))
            continue
        curve = step_function_lookup(x_array.tolist(), frac.tolist(), x_grid)
        # Grid cells before a run's first logged point come back as the +inf
        # sentinel `step_function_lookup` uses; for an ECDF that means zero
        # targets reached, not infinity.
        rows.append(np.where(np.isinf(curve), 0.0, curve))
    if not rows:
        return np.zeros(x_grid.shape, dtype=np.float64)
    if empty:
        warnings.warn(
            f"{empty} of {len(rows)} runs had no {field}/{x_field} data and "
            f"contribute all-zero rows, lowering the mean ECDF",
            RuntimeWarning,
            stacklevel=2,
        )
    return np.mean(rows, axis=0)


def ecdf_auc(x_grid: NDArray[np.int64], curve: NDArray[np.float64]) -> float:
    """Area under an ECDF curve, normalised to ``[0, 1]``.

    Integrated against ``log10(x_grid)``, because the curve is drawn on a
    log budget axis: a linear integral is dominated by the last decade, so
    two algorithms two decades apart in speed score almost the same and the
    number stops matching what the plot shows.

    Returns ``0.0`` for a grid too short to integrate over.
    """
    if x_grid.size < 2:
        return 0.0
    log_x = np.log10(np.asarray(x_grid, dtype=np.float64))
    span = float(log_x[-1] - log_x[0])
    if span <= 0.0:
        return 0.0
    return float(trapezoid(curve, log_x) / span)
