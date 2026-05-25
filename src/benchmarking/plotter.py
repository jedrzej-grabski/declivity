"""Plotting helpers for the benchmarking framework.

Produces convergence plots with median + IQR bands across seeds, plus
final-fitness distribution boxplots. Designed to make the experiment's
key claim — "this combo beats either alone" — readable at a glance.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.benchmarking.aggregation import (
    common_evaluation_grid,
    percentile_band,
    stack_traces_on_grid,
)
from src.benchmarking.algorithm_run import AlgorithmRun
from src.benchmarking.problem import Problem
from src.benchmarking.run_trace import RunTrace


class BenchmarkPlotter:
    """Plotter for results coming out of a Benchmark run."""

    def __init__(
        self,
        problems: list[Problem],
        algorithms: list[AlgorithmRun],
        traces: dict[tuple[str, str], list[RunTrace]],
        output_dir: Union[str, Path],
        floor: float = 1e-12,
    ):
        self.problems = problems
        self.algorithms = algorithms
        self.traces = traces
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.floor = floor

    def _algorithm_color(self, algorithm: AlgorithmRun) -> str:
        return algorithm.color

    def plot_convergence_grid(
        self,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        num_grid_points: int = 200,
        show_iqr: bool = True,
        linewidth: float = 2.5,
        iqr_alpha: float = 0.15,
        legend_fontsize: int = 10,
        figsize_per_panel: tuple[float, float] = (8.5, 6.0),
        annotate_handoff: bool = True,
    ) -> plt.Figure:
        """One subplot per problem. Median + IQR across seeds for each algorithm.

        Floors fitness values at ``self.floor`` before log-scaling so the
        y-axis remains finite for runs that converge essentially to zero.
        """
        num_problems = len(self.problems)
        cols = min(num_problems, 2)
        rows = (num_problems + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        )
        axes = np.atleast_1d(axes).flatten()

        for problem_idx, problem in enumerate(self.problems):
            ax = axes[problem_idx]

            # Build a shared grid spanning all algorithm traces for this problem.
            all_traces: list[RunTrace] = []
            for algorithm in self.algorithms:
                all_traces.extend(self.traces.get((problem.name, algorithm.name), []))
            if not all_traces:
                ax.set_visible(False)
                continue
            grid = common_evaluation_grid(all_traces, num_points=num_grid_points)

            handoff_markers: dict[int, Optional[int]] = {}
            """eval at handoff -> iter at handoff (if known) for this panel."""

            for algorithm in self.algorithms:
                traces = self.traces.get((problem.name, algorithm.name), [])
                if not traces:
                    continue

                matrix = stack_traces_on_grid(traces, grid)
                matrix = np.maximum(matrix, self.floor)
                median, low, high = percentile_band(matrix)

                color = self._algorithm_color(algorithm)
                final_median = float(np.median([t.final_fitness for t in traces]))
                ax.semilogy(
                    grid,
                    median,
                    color=color,
                    linewidth=linewidth,
                    label=f"{algorithm.name}  (median = {final_median:.2e})",
                )
                if show_iqr and len(traces) > 1:
                    ax.fill_between(grid, low, high, color=color, alpha=iqr_alpha)

                handoff_traces = [t for t in traces if t.handoff_eval is not None]
                if handoff_traces:
                    median_eval = int(np.median([t.handoff_eval for t in handoff_traces]))
                    iters = [t.handoff_iter for t in handoff_traces if t.handoff_iter is not None]
                    median_iter = int(np.median(iters)) if iters else None
                    handoff_markers[median_eval] = median_iter

            for handoff_eval, handoff_iter in handoff_markers.items():
                ax.axvline(
                    handoff_eval,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.45,
                )
                if annotate_handoff:
                    if handoff_iter is not None:
                        label = f"  handoff @ {handoff_iter} gen ({handoff_eval} evals)"
                    else:
                        label = f"  handoff @ {handoff_eval} evals"
                    ymax = ax.get_ylim()[1]
                    ax.text(
                        handoff_eval, ymax,
                        label,
                        rotation=90, va="top", ha="left",
                        fontsize=8, color="gray", alpha=0.7,
                    )

            ax.set_xlabel("Function Evaluations", fontsize=12)
            ax.set_ylabel("Best Fitness (log)", fontsize=12)
            ax.set_title(
                f"{problem.name}  (d={problem.dimensions}, "
                f"n_seeds={len(traces)})",
                fontsize=13,
            )
            ax.grid(True, alpha=0.25, which="both")
            ax.legend(fontsize=legend_fontsize, loc="best", framealpha=0.9)
            ax.tick_params(axis="both", labelsize=10)

        for idx in range(num_problems, len(axes)):
            axes[idx].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=14, y=1.01)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=110, bbox_inches="tight")
        return fig

    def plot_final_fitness_boxplot(
        self,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Boxplot of final fitness per algorithm, one panel per problem.

        Runs that reached zero or sub-floor fitness are dropped from the
        boxplot (a log y-axis can't display them honestly). The number of
        such runs per algorithm is annotated above the box as ``n=*/N``
        where the star is the surviving count.
        """
        num_problems = len(self.problems)
        cols = min(num_problems, 2)
        rows = (num_problems + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))
        axes = np.atleast_1d(axes).flatten()

        for problem_idx, problem in enumerate(self.problems):
            ax = axes[problem_idx]

            data: list[NDArray[np.float64]] = []
            labels: list[str] = []
            colors: list[str] = []
            dropped_counts: list[tuple[int, int]] = []
            for algorithm in self.algorithms:
                traces = self.traces.get((problem.name, algorithm.name), [])
                if not traces:
                    continue
                raw = np.array([t.final_fitness for t in traces])
                surviving = raw[raw > self.floor]
                if surviving.size == 0:
                    # Everything reached zero; nothing to plot. Skip but
                    # still record so the label/count appears.
                    data.append(np.array([self.floor]))
                else:
                    data.append(surviving)
                labels.append(algorithm.name)
                colors.append(self._algorithm_color(algorithm))
                dropped_counts.append((int(surviving.size), int(raw.size)))

            if not data:
                ax.set_visible(False)
                continue

            box = ax.boxplot(
                data,
                labels=labels,
                patch_artist=True,
                showmeans=True,
                meanline=True,
            )
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            ax.set_yscale("log")
            ax.set_ylabel("Final Best Fitness (log)", fontsize=11)
            ax.set_title(
                f"{problem.name} (d={problem.dimensions})",
                fontsize=12,
            )
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis="x", labelrotation=15)

            for tick_idx, (kept, total) in enumerate(dropped_counts, start=1):
                if kept < total:
                    ax.annotate(
                        f"n={kept}/{total}",
                        xy=(tick_idx, 0.98),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="top",
                        fontsize=8, color="dimgray",
                    )

        for idx in range(num_problems, len(axes)):
            axes[idx].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=14, y=1.01)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=110, bbox_inches="tight")
        return fig
