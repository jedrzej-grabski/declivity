"""Multi-seed declarative plotting.

Counterpart to :py:mod:`src.plotting.declarative` for the multi-seed,
multi-problem case. Where ``plot_metrics`` and ``plot_comparison`` operate
on ``OptimizationResult``s (full diagnostic logs from a single run), the
functions here operate on ``RunTrace`` lists — the lean per-seed records
that the benchmarking framework persists.

The grid layout is one panel per problem (rather than one panel per
metric, like the single-run plotter), because for multi-seed work the
interesting comparison axis is the problem. Inside each panel,
algorithms are overlaid with a median line and an IQR band.

For now only convergence (best fitness vs. evaluations) and final-fitness
distributions are supported — those are the only quantities ``RunTrace``
carries across seeds. If diagnostics-per-seed ever land on ``RunTrace``,
extending the system is a matter of adding panel-aware logic here.
"""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.benchmarking.aggregation import (
    common_evaluation_grid,
    percentile_band,
    stack_traces_on_grid,
)
from src.benchmarking.algorithm_run import AlgorithmRun
from src.benchmarking.problem import Problem
from src.benchmarking.run_trace import RunTrace


def _grid_dims(num_panels: int, ncols: int) -> tuple[int, int]:
    cols = max(1, min(ncols, num_panels))
    rows = (num_panels + cols - 1) // cols
    return rows, cols


def _save_if_path(fig: Figure, save_path: Path | str | None) -> None:
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


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
        traces: ``{(problem.name, algorithm.name): [RunTrace per seed]}``.
            Typically the return value of ``Benchmark.run()``.
        problems: Outer iteration order — one subplot per problem in this
            order. Provides ``dimensions`` for sub-titles.
        algorithms: Inner iteration order — algorithms are overlaid in
            each subplot in this order. ``color`` and ``name`` from each
            ``AlgorithmRun`` are used directly.
        ncols: Subplots per row.
        num_grid_points: Resolution of the shared evaluation grid used
            to align step-function traces. Log-spaced by default since
            most convergence happens early.
        show_iqr: Draw the 25/75 percentile band beneath the median.
        iqr_alpha: Transparency of the IQR fill.
        floor: Clip values below this before log-scaling — runs that
            reach effectively zero would otherwise blow up.
        annotate_handoff: Add a vertical line and label at the median
            handoff evaluation, when any trace reports one.
        figsize_per_panel: ``(width, height)`` in inches per subplot.
        linewidth: Line width for the median curves.
        legend_fontsize: Legend text size.
        title: Optional figure-level suptitle.
        save_path: If set, the figure is saved here.

    Returns:
        The matplotlib :class:`Figure`.
    """
    problems_list = list(problems)
    algorithms_list = list(algorithms)
    if not problems_list:
        raise ValueError("problems must be non-empty")

    rows, cols = _grid_dims(len(problems_list), ncols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for panel_index, problem in enumerate(problems_list):
        ax = flat_axes[panel_index]

        # Shared evaluation grid spanning every trace on this problem.
        all_traces: list[RunTrace] = []
        for algorithm in algorithms_list:
            all_traces.extend(traces.get((problem.name, algorithm.name), []))
        if not all_traces:
            ax.set_visible(False)
            continue
        grid = common_evaluation_grid(all_traces, num_points=num_grid_points)

        # eval -> (iter | None) for handoff markers seen on this panel.
        # Keyed by eval so multiple traces with the same handoff collapse
        # into one annotation; the median per algorithm is what we draw.
        handoff_markers: dict[int, int | None] = {}
        traces_count = 0

        for algorithm in algorithms_list:
            algorithm_traces = traces.get((problem.name, algorithm.name), [])
            if not algorithm_traces:
                continue
            traces_count = max(traces_count, len(algorithm_traces))

            matrix = stack_traces_on_grid(algorithm_traces, grid)
            matrix = np.maximum(matrix, floor)
            median, low, high = percentile_band(matrix)

            final_median = float(
                np.median([trace.final_fitness for trace in algorithm_traces])
            )
            ax.semilogy(
                grid,
                median,
                color=algorithm.color,
                linewidth=linewidth,
                label=f"{algorithm.name}  (median = {final_median:.2e})",
            )
            if show_iqr and len(algorithm_traces) > 1:
                ax.fill_between(grid, low, high, color=algorithm.color, alpha=iqr_alpha)

            handoff_traces = [
                trace for trace in algorithm_traces if trace.handoff_eval is not None
            ]
            if handoff_traces:
                handoff_evals = [
                    e for trace in handoff_traces
                    if (e := trace.handoff_eval) is not None
                ]
                median_eval = int(np.median(handoff_evals))
                handoff_iters = [
                    trace.handoff_iter
                    for trace in handoff_traces
                    if trace.handoff_iter is not None
                ]
                handoff_markers[median_eval] = (
                    int(np.median(handoff_iters)) if handoff_iters else None
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

    _save_if_path(fig, save_path)
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

    Runs that reached sub-``floor`` fitness are dropped (they can't be
    shown honestly on a log axis); the count of surviving runs is
    annotated as ``n=X/Y`` above each box when ``X < Y``.

    Args:
        traces: ``{(problem.name, algorithm.name): [RunTrace per seed]}``.
        problems: One subplot per problem in this order.
        algorithms: Algorithms appear as boxes left-to-right in this order.
        ncols, floor, figsize_per_panel, title, save_path: As in
            :py:func:`plot_benchmark_convergence`.

    Returns:
        The matplotlib :class:`Figure`.
    """
    problems_list = list(problems)
    algorithms_list = list(algorithms)
    if not problems_list:
        raise ValueError("problems must be non-empty")

    rows, cols = _grid_dims(len(problems_list), ncols)
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
            if surviving.size == 0:
                boxes.append(np.array([floor]))
            else:
                boxes.append(surviving)
            labels.append(algorithm.name)
            colors.append(algorithm.color)
            surviving_counts.append((int(surviving.size), int(raw.size)))

        if not boxes:
            ax.set_visible(False)
            continue

        box_artists = ax.boxplot(
            boxes,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
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

    _save_if_path(fig, save_path)
    return fig
