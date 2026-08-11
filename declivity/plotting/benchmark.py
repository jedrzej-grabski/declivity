"""Multi-seed benchmark plotting: median + IQR bands over ``RunTrace`` lists.

The multi-seed, multi-problem counterpart to
:py:mod:`declivity.plotting.declarative`.  These functions lay out the grid
(one panel per problem, or a single overlay axes) and draw the curves through
the same :func:`~declivity.plotting.unified.draw_groups` core the single-run
plotters use.

Convergence (best fitness vs. evaluations), final-fitness distributions and
target-hitting ECDFs are first-class here, since every ``RunTrace`` carries
them.  Retained scalar series (``sigma``, ...) can be banded across seeds via
:func:`~declivity.plotting.unified.plot_panels` with the matching panel key.

The ECDF entry point takes a single ``problem`` rather than an iterable: it
collapses one problem's runs into one curve per algorithm.  The suite-wide
COCO figure, pooling (problem, target) pairs across a function family, is a
different aggregation and is not built here.
"""

import warnings
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from declivity.benchmarking.aggregation import series_grid
from declivity.benchmarking.algorithm_run import AlgorithmRun
from declivity.benchmarking.ecdf import (
    DEFAULT_THRESHOLD_FLOOR,
    aggregate_ecdf,
    ecdf_auc,
    threshold_grid,
)
from declivity.benchmarking.problem import Problem
from declivity.benchmarking.run_trace import RunTrace
from declivity.plotting.unified import (
    RunGroup,
    draw_groups,
    grid_dims,
    save_if_path,
    thin_linear_grid,
    thin_log_grid,
)


def plot_benchmark_convergence(
    traces: dict[tuple[str, str], list[RunTrace]],
    problems: Iterable[Problem],
    algorithms: Iterable[AlgorithmRun],
    *,
    ncols: int = 2,
    num_grid_points: int = 200,
    show_iqr: bool = True,
    iqr_alpha: float = 0.15,
    floor: float = 1e-12,
    annotate_handoff: bool = True,
    figsize_per_panel: tuple[float, float] = (8.5, 6.0),
    linewidth: float = 2.5,
    legend_fontsize: int = 10,
    title: str | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """One panel per problem, every algorithm overlaid with median + IQR.

    Args:
        traces: ``{(problem.name, algorithm.name): [RunTrace per seed]}`` —
            typically the return value of ``Benchmark.run()``.
        problems: Outer iteration order — one subplot per problem.
        algorithms: Inner iteration order — algorithms overlaid per subplot;
            ``color`` / ``name`` are read from each ``AlgorithmRun``.
        ncols: Subplots per row.
        num_grid_points: Resolution of the shared evaluation grid.
        show_iqr: Draw the 25/75 percentile band beneath the median.
        iqr_alpha: Transparency of the IQR fill.
        floor: Clip values below this before log-scaling.
        annotate_handoff: Add a vertical line + label at the median handoff
            evaluation, when any trace reports one.
        figsize_per_panel, linewidth, legend_fontsize, title, save_path:
            cosmetics / output.

    Returns:
        The matplotlib :class:`Figure`.
    """
    problems_list = list(problems)
    algorithms_list = list(algorithms)
    if not problems_list:
        raise ValueError("problems must be non-empty")

    rows, cols = grid_dims(len(problems_list), ncols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for panel_index, problem in enumerate(problems_list):
        ax = flat_axes[panel_index]

        groups: list[RunGroup] = []
        traces_count = 0
        # eval -> (iter | None) for handoff markers seen on this panel.
        handoff_markers: dict[int, int | None] = {}

        for algorithm in algorithms_list:
            algorithm_traces = traces.get((problem.name, algorithm.name), [])
            if not algorithm_traces:
                continue
            traces_count = max(traces_count, len(algorithm_traces))
            groups.append(
                RunGroup.from_runs(
                    algorithm.name, algorithm_traces, color=algorithm.color
                )
            )

            handoff_traces = [t for t in algorithm_traces if t.handoff_eval is not None]
            if handoff_traces:
                handoff_evals = [
                    e for t in handoff_traces if (e := t.handoff_eval) is not None
                ]
                median_eval = int(np.median(handoff_evals))
                handoff_iters = [
                    t.handoff_iter for t in handoff_traces if t.handoff_iter is not None
                ]
                handoff_markers[median_eval] = (
                    int(np.median(handoff_iters)) if handoff_iters else None
                )

        if not groups:
            ax.set_visible(False)
            continue

        # The shared renderer: median + IQR per algorithm, one curve each.
        draw_groups(
            ax,
            groups,
            field="best_fitness",
            x_field="evaluations",
            floor=floor,
            aggregate=True,
            show_band=show_iqr,
            iqr_alpha=iqr_alpha,
            linewidth=linewidth,
            num_grid_points=num_grid_points,
            annotate_final="median",
        )

        if annotate_handoff:
            for handoff_eval, handoff_iter in handoff_markers.items():
                ax.axvline(
                    handoff_eval,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.45,
                )
                if handoff_iter is not None:
                    label = f"  handoff @ {handoff_iter} gen ({handoff_eval} evals)"
                else:
                    label = f"  handoff @ {handoff_eval} evals"
                ymax = ax.get_ylim()[1]
                ax.text(
                    handoff_eval,
                    ymax,
                    label,
                    rotation=90,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="gray",
                    alpha=0.7,
                )

        ax.set_xlabel("Function Evaluations", fontsize=12)
        ax.set_ylabel("Best Fitness (log)", fontsize=12)
        ax.set_yscale("log")
        ax.set_title(
            f"{problem.name}  (d={problem.dimensions}, n_seeds={traces_count})",
            fontsize=13,
        )
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=legend_fontsize, loc="best", framealpha=0.9)
        ax.tick_params(axis="both", labelsize=10)

    for panel_index in range(len(problems_list), len(flat_axes)):
        flat_axes[panel_index].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()

    save_if_path(fig, save_path)
    return fig


def plot_benchmark_boxplot(
    traces: dict[tuple[str, str], list[RunTrace]],
    problems: Iterable[Problem],
    algorithms: Iterable[AlgorithmRun],
    *,
    ncols: int = 2,
    floor: float = 1e-12,
    figsize_per_panel: tuple[float, float] = (7.0, 4.5),
    title: str | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """Final-fitness distribution per algorithm, one panel per problem.

    Runs that reached sub-``floor`` fitness are dropped, since a log axis
    cannot show them; the count of surviving runs is annotated as ``n=X/Y``
    above each box when ``X < Y``.

    This view is a final scalar rather than a time series, so it does not go
    through ``draw_groups``, but it reads the same ``RunTrace`` records and
    per-algorithm colours.
    """
    problems_list = list(problems)
    algorithms_list = list(algorithms)
    if not problems_list:
        raise ValueError("problems must be non-empty")

    rows, cols = grid_dims(len(problems_list), ncols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for panel_index, problem in enumerate(problems_list):
        ax = flat_axes[panel_index]

        boxes: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[str] = []
        surviving_counts: list[tuple[int, int]] = []

        for algorithm in algorithms_list:
            algorithm_traces = traces.get((problem.name, algorithm.name), [])
            if not algorithm_traces:
                continue
            raw = np.array([trace.final_fitness for trace in algorithm_traces])
            surviving = raw[raw > floor]
            boxes.append(surviving if surviving.size else np.array([floor]))
            labels.append(algorithm.name)
            colors.append(algorithm.color)
            surviving_counts.append((int(surviving.size), int(raw.size)))

        if not boxes:
            ax.set_visible(False)
            continue

        # ``tick_labels`` replaced ``labels`` in Matplotlib 3.9; the old
        # spelling was removed in 3.11, which is the version this repo pins.
        box_artists = ax.boxplot(
            boxes, tick_labels=labels, patch_artist=True, showmeans=True, meanline=True
        )
        for patch, color in zip(box_artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_yscale("log")
        ax.set_ylabel("Final Best Fitness (log)", fontsize=11)
        ax.set_title(f"{problem.name} (d={problem.dimensions})", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelrotation=15)

        for tick_index, (kept, total) in enumerate(surviving_counts, start=1):
            if kept < total:
                ax.annotate(
                    f"n={kept}/{total}",
                    xy=(tick_index, 0.98),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="dimgray",
                )

    for panel_index in range(len(problems_list), len(flat_axes)):
        flat_axes[panel_index].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()

    save_if_path(fig, save_path)
    return fig


def plot_convergence_overlay(
    traces: dict[tuple[str, str], list[RunTrace]],
    problem: Problem,
    algorithms: Iterable[AlgorithmRun],
    *,
    title: str | None = None,
    xlabel: str = "Function Evaluations",
    ylabel: str = "Best Fitness (log)",
    floor: float = 1e-12,
    show_iqr: bool = True,
    iqr_alpha: float = 0.15,
    num_grid_points: int = 300,
    annotate_final: bool = True,
    xmax: float | None = None,
    secondary_iter_lambda: int | None = None,
    secondary_label: str = "CMA-ES iterations",
    secondary_location: float = -0.16,
    linewidth: float = 2.2,
    legend_loc: str = "best",
    legend_fontsize: int = 9,
    figsize: tuple[float, float] = (9.0, 6.2),
    save_path: Path | str | None = None,
) -> Figure:
    """Single-panel overlay: one convergence curve per algorithm, on one problem.

    The line-per-algorithm counterpart of :func:`plot_benchmark_convergence`
    (which lays out one panel *per problem*). Use this when the interesting
    axis is the *algorithm* — a multi-``k`` sweep on a single problem, or any
    single-seed comparison — optionally with a secondary "iterations" axis.

    With one seed per algorithm the raw trace is drawn; with several, the
    median across seeds (plus an optional IQR band), via the shared
    ``draw_groups`` core. Colours / names come from the ``AlgorithmRun``
    objects, drawn in the given order (last on top).

    Args:
        traces: ``{(problem.name, algorithm.name): [RunTrace per seed]}``.
        problem: The single problem to plot.
        algorithms: Draw order; ``color`` / ``name`` read from each.
        secondary_iter_lambda: If set, add a secondary x-axis showing
            iterations ``= evaluations / (lambda + 1)``. ``None`` omits it.
        annotate_final: Append the (median) final fitness to each legend label.
        xmax: Clip the evaluation axis here — trims flat converged tails that
            would otherwise compress the interesting region.
        save_path: If set, the figure is written here.

    Returns:
        The matplotlib :class:`Figure`.
    """
    groups = [
        RunGroup.from_runs(
            algorithm.name,
            traces.get((problem.name, algorithm.name), []),
            color=algorithm.color,
        )
        for algorithm in algorithms
    ]

    fig, ax = plt.subplots(figsize=figsize)
    draw_groups(
        ax,
        groups,
        field="best_fitness",
        x_field="evaluations",
        floor=floor,
        aggregate=True,
        show_band=show_iqr,
        iqr_alpha=iqr_alpha,
        linewidth=linewidth,
        num_grid_points=num_grid_points,
        annotate_final="median" if annotate_final else None,
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_yscale("log")
    if xmax is not None:
        ax.set_xlim(left=0.0, right=xmax)
    thin_log_grid(ax, "y", max_ticks=7)
    thin_linear_grid(ax, "x", max_ticks=6)
    ax.grid(True, alpha=0.18, linewidth=0.7, which="major")
    # legend_loc stays a plain str in this signature; matplotlib 3.11 narrowed
    # loc to a Literal union, which a str cannot satisfy statically.
    ax.legend(fontsize=legend_fontsize, loc=legend_loc, framealpha=0.9)  # pyright: ignore[reportArgumentType]
    ax.tick_params(axis="both", labelsize=10)

    if secondary_iter_lambda is not None:
        evals_per_iter = float(secondary_iter_lambda + 1)

        def evals_to_iters(evaluations: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(evaluations, dtype=float) / evals_per_iter

        def iters_to_evals(iterations: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(iterations, dtype=float) * evals_per_iter

        secondary = ax.secondary_xaxis(
            secondary_location,
            functions=(evals_to_iters, iters_to_evals),  # type: ignore[arg-type]
        )
        secondary.set_xlabel(secondary_label, fontsize=12)

    fig.tight_layout()
    save_if_path(fig, save_path)
    return fig


def _problem_optimum(problem: Problem) -> float:
    """``f*`` for ``problem``, or 0.0 with a warning if it publishes none.

    ``BenchmarkFunction.global_minimum`` is the pair ``(x*, f*)``, so only
    the second element is wanted. CEC problems report a nonzero ``f*`` (the
    ``100·i`` bias) and ``x*`` as NaN, which is why the value is taken
    positionally and checked for finiteness rather than trusted wholesale.
    """
    minimum = getattr(problem.function, "global_minimum", None)
    if minimum is not None:
        try:
            value = float(minimum[1])
        except (TypeError, IndexError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            return value
    warnings.warn(
        f"{problem.name!r} publishes no finite global minimum; thresholding "
        f"raw fitness as if f*=0. Pass global_minimum= explicitly, or "
        f"gap_to_optimum=False to use raw fitness.",
        RuntimeWarning,
        stacklevel=3,
    )
    return 0.0


def plot_suite_ecdf(
    traces: dict[tuple[str, str], list[RunTrace]],
    problems: Iterable[Problem],
    algorithms: Iterable[AlgorithmRun],
    *,
    n_thresholds: int = 50,
    num_grid_points: int = 200,
    threshold_floor: float = DEFAULT_THRESHOLD_FLOOR,
    gap_to_optimum: bool = True,
    annotate_auc: bool = True,
    title: str | None = None,
    show_subtitle: bool = True,
    xlabel: str = "Function Evaluations",
    ylabel: str = "Fraction of targets reached",
    linewidth: float = 2.2,
    legend_loc: str = "best",
    legend_fontsize: int = 9,
    figsize: tuple[float, float] = (9.0, 6.2),
    save_path: Path | str | None = None,
) -> Figure:
    """Aggregated ECDF over a whole problem suite: one curve per algorithm.

    The COCO-style figure: each problem contributes its own log-spaced target
    grid (gap to its own ``f*``, pooled over every algorithm's runs on that
    problem), every ``(problem, target)`` pair weighs equally, and an
    algorithm's curve is the mean over problems of its per-problem mean ECDF
    on one shared evaluation-budget grid.  The single-problem view is
    :func:`plot_benchmark_ecdf`.
    """
    problems_list = list(problems)
    algorithms_list = list(algorithms)
    if not problems_list:
        raise ValueError("problems must be non-empty")

    pooled_all = [
        trace
        for problem in problems_list
        for algorithm in algorithms_list
        for trace in traces.get((problem.name, algorithm.name), [])
    ]
    if not pooled_all:
        raise ValueError("No traces found for the given problems/algorithms")
    x_grid = series_grid(pooled_all, "evaluations", num_grid_points)

    per_problem: list[tuple[Problem, float, NDArray[np.float64]]] = []
    for problem in problems_list:
        pooled = [
            trace
            for algorithm in algorithms_list
            for trace in traces.get((problem.name, algorithm.name), [])
        ]
        if not pooled:
            continue
        optimum = _problem_optimum(problem) if gap_to_optimum else 0.0
        thresholds = threshold_grid(
            pooled,
            n_thresholds=n_thresholds,
            floor=threshold_floor,
            global_minimum=optimum,
            gap_to_optimum=gap_to_optimum,
        )
        per_problem.append((problem, optimum, thresholds))

    fig, ax = plt.subplots(figsize=figsize)
    seed_counts: list[int] = []
    for algorithm in algorithms_list:
        curves: list[NDArray[np.float64]] = []
        for problem, optimum, thresholds in per_problem:
            algorithm_traces = traces.get((problem.name, algorithm.name), [])
            if not algorithm_traces:
                continue
            seed_counts.append(len(algorithm_traces))
            curves.append(
                aggregate_ecdf(
                    algorithm_traces,
                    thresholds,
                    x_grid,
                    global_minimum=optimum,
                    gap_to_optimum=gap_to_optimum,
                )
            )
        if not curves:
            continue
        curve = np.mean(curves, axis=0)
        label = algorithm.name
        if annotate_auc:
            label = f"{label} (AUC={ecdf_auc(x_grid, curve):.3f})"
        ax.plot(x_grid, curve, label=label, color=algorithm.color, linewidth=linewidth)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0.0, 1.02)
    subtitle = (
        f"{len(per_problem)} problems, "
        f"n_seeds={max(seed_counts) if seed_counts else 0}, "
        f"{n_thresholds} targets per problem"
        f"{'' if gap_to_optimum else ', raw fitness'}"
    )
    if show_subtitle:
        ax.set_title(f"{title}\n{subtitle}" if title else subtitle, fontsize=12)
    elif title:
        ax.set_title(title, fontsize=12)
    thin_log_grid(ax, "x")
    ax.grid(True, alpha=0.18, linewidth=0.7, which="major")
    ax.legend(fontsize=legend_fontsize, loc=legend_loc, framealpha=0.9)  # pyright: ignore[reportArgumentType]
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()

    save_if_path(fig, save_path)
    return fig


def plot_benchmark_ecdf(
    traces: dict[tuple[str, str], list[RunTrace]],
    problem: Problem,
    algorithms: Iterable[AlgorithmRun],
    *,
    n_thresholds: int = 50,
    num_grid_points: int = 200,
    threshold_floor: float = DEFAULT_THRESHOLD_FLOOR,
    threshold_ceiling: float | None = None,
    global_minimum: float | None = None,
    gap_to_optimum: bool = True,
    annotate_auc: bool = True,
    title: str | None = None,
    xlabel: str = "Function Evaluations",
    ylabel: str = "Fraction of targets reached",
    linewidth: float = 2.2,
    legend_loc: str = "best",
    legend_fontsize: int = 9,
    figsize: tuple[float, float] = (9.0, 6.2),
    save_path: Path | str | None = None,
) -> Figure:
    """Single-problem ECDF overlay: one curve per algorithm.

    Pools every algorithm's runs on ``problem`` to build one shared
    threshold grid (see :func:`~declivity.benchmarking.ecdf.threshold_grid`),
    so every algorithm is judged against identical targets, then discretises
    each algorithm's mean ECDF onto a shared evaluation-budget grid spanning
    every run's own evaluation counts.

    Args:
        traces: ``{(problem.name, algorithm.name): [RunTrace per seed]}``.
        problem: The single problem to plot.
        algorithms: Draw order; ``color`` / ``name`` read from each.
        n_thresholds: Number of log-spaced target levels.
        num_grid_points: Resolution of the shared evaluation-budget grid.
        threshold_ceiling: Fix the top of the target range instead of taking
            it from the pooled data. Needed for curves that have to be
            comparable across separate figures.
        global_minimum: ``problem``'s known ``f*``, subtracted from
            ``best_fitness`` before thresholding (see
            :mod:`~declivity.benchmarking.ecdf`). ``None`` (default) reads it
            off ``problem.function.global_minimum``, falling back to 0.0 with
            a warning when the objective does not publish one. Passing 0.0
            explicitly is only right for a problem whose optimum really is
            zero — on the CEC suite ``f*`` is a nonzero bias, and getting it
            wrong makes every target below ``f*`` unreachable, so all curves
            plateau together and the figure discriminates nothing. Ignored
            when ``gap_to_optimum=False``.
        gap_to_optimum: Whether to subtract ``global_minimum`` (default) or
            threshold raw ``best_fitness`` values.
        annotate_auc: Append each curve's normalised AUC to its legend label.
            Integrated in the log budget domain, matching the drawn axis.
        save_path: If set, the figure is written here (dpi 150).

    Returns:
        The matplotlib :class:`Figure`.
    """
    algorithms_list = list(algorithms)
    pooled_traces = [
        trace
        for algorithm in algorithms_list
        for trace in traces.get((problem.name, algorithm.name), [])
    ]
    if not pooled_traces:
        raise ValueError(f"No traces found for problem {problem.name!r}")

    if global_minimum is None:
        global_minimum = _problem_optimum(problem) if gap_to_optimum else 0.0

    thresholds = threshold_grid(
        pooled_traces,
        n_thresholds=n_thresholds,
        floor=threshold_floor,
        global_minimum=global_minimum,
        gap_to_optimum=gap_to_optimum,
        ceiling=threshold_ceiling,
    )
    x_grid = series_grid(pooled_traces, "evaluations", num_grid_points)

    fig, ax = plt.subplots(figsize=figsize)
    seed_counts: list[int] = []
    for algorithm in algorithms_list:
        algorithm_traces = traces.get((problem.name, algorithm.name), [])
        if not algorithm_traces:
            continue
        seed_counts.append(len(algorithm_traces))
        curve = aggregate_ecdf(
            algorithm_traces,
            thresholds,
            x_grid,
            global_minimum=global_minimum,
            gap_to_optimum=gap_to_optimum,
        )
        label = algorithm.name
        if annotate_auc:
            label = f"{label} (AUC={ecdf_auc(x_grid, curve):.3f})"
        ax.plot(x_grid, curve, label=label, color=algorithm.color, linewidth=linewidth)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0.0, 1.02)
    # Record which targets, dimension and seed count produced the curves.
    subtitle = (
        f"{problem.name}  (d={problem.dimensions}, "
        f"n_seeds={max(seed_counts) if seed_counts else 0}, "
        f"{n_thresholds} targets in [{thresholds[0]:.1e}, {thresholds[-1]:.1e}]"
        f"{'' if gap_to_optimum else ', raw fitness'})"
    )
    ax.set_title(f"{title}\n{subtitle}" if title else subtitle, fontsize=12)
    thin_log_grid(ax, "x")
    ax.grid(True, alpha=0.18, linewidth=0.7, which="major")
    # See plot_convergence_overlay: matplotlib 3.11 narrowed loc to a Literal
    # union that a plain str cannot satisfy statically.
    ax.legend(fontsize=legend_fontsize, loc=legend_loc, framealpha=0.9)  # pyright: ignore[reportArgumentType]
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()

    save_if_path(fig, save_path)
    return fig
