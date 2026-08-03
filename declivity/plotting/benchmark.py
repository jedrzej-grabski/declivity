"""Multi-seed benchmark plotting — median + IQR bands over ``RunTrace`` lists.

Counterpart to :py:mod:`src.plotting.declarative` for the multi-seed,
multi-problem case. These functions lay out the grid (one panel per problem,
or a single overlay axes) and draw the curves through the *same*
:func:`~src.plotting.unified.draw_groups` core the single-run plotters use —
so a benchmark of one seed renders as a line and a benchmark of 25 renders as
a median + IQR band, through one code path.

For now only convergence (best fitness vs. evaluations) and final-fitness
distributions are first-class here, because those are the quantities every
``RunTrace`` carries. Any *retained* scalar series (``sigma``, ...) can be
banded across seeds via :func:`~src.plotting.unified.plot_panels` with the
matching panel key.
"""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from declivity.benchmarking.algorithm_run import AlgorithmRun
from declivity.benchmarking.problem import Problem
from declivity.benchmarking.run_trace import RunTrace
from declivity.plotting.unified import (
    RunGroup,
    draw_groups,
    grid_dims,
    save_if_path,
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

    Runs that reached sub-``floor`` fitness are dropped (they can't be shown
    honestly on a log axis); the count of surviving runs is annotated as
    ``n=X/Y`` above each box when ``X < Y``.

    This view is final-*scalar*, not a time series, so it doesn't go through
    ``draw_groups`` — but it reads the same ``RunTrace`` records and the same
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

        box_artists = ax.boxplot(
            boxes, labels=labels, patch_artist=True, showmeans=True, meanline=True
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
        save_path: If set, the figure is written here (dpi 150).

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
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=legend_fontsize, loc=legend_loc, framealpha=0.9)
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
