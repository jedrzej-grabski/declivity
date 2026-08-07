"""Staircase plot for the interleaved CMA-ES <-> L-BFGS-B scheme.

A single-run diagnostic outside the panel/registry system.  It overlays
three curves derived from one
:class:`~declivity.benchmarking.algorithm_run.InterleaveResult`:

- the overall-best staircase: monotone, combining CMA-ES progress with each
  L-BFGS-B drop;
- the CMA-ES backbone: best-so-far over CMA-ES generations only, which stays
  above the overall best between bursts;
- the L-BFGS-B bursts: the sharp drops, coloured separately.

An optional standalone-CMA-ES baseline can be overlaid.  With a shared seed
it is the same trajectory as the backbone, stretched along the evaluation
axis by the budget the probes consume.

Where :func:`declivity.plotting.plot_benchmark_convergence` shows medians
across seeds, this dissects a single run.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from declivity.benchmarking.algorithm_run import InterleaveResult
from declivity.benchmarking.run_trace import RunTrace


def _save_if_path(fig: Figure, save_path: Path | str | None) -> None:
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_interleaved_convergence(
    result: InterleaveResult,
    *,
    title: str | None = None,
    overall_label: str = "Interleaved — overall best",
    overall_color: str = "#e67e22",
    cmaes_label: str = "CMA-ES backbone",
    cmaes_color: str = "#27ae60",
    burst_label: str = "L-BFGS-B bursts",
    burst_color: str = "#e74c3c",
    baseline_trace: RunTrace | None = None,
    baseline_label: str = "CMA-ES (standalone)",
    baseline_color: str = "#7f8c8d",
    floor: float = 1e-12,
    mark_intervals: bool = True,
    figsize: tuple[float, float] = (10.0, 6.5),
    save_path: Path | str | None = None,
) -> Figure:
    """Render the interleaved-run staircase.

    Args:
        result: The :class:`InterleaveResult` from
            :py:meth:`InterleavedCMAESLBFGSB.run_with_detail`.
        baseline_trace: Optional standalone-CMA-ES :class:`RunTrace` (same
            problem/seed) to overlay as a faint reference.
        floor: Values clipped to this before log-scaling.
        mark_intervals: Draw a faint vertical line where each burst begins.
        save_path: If set, the figure is written here (dpi 150).

    Returns:
        The matplotlib :class:`Figure`.
    """
    fig, ax = plt.subplots(figsize=figsize)

    def clip(values: list[float]) -> np.ndarray:
        return np.maximum(np.asarray(values, dtype=float), floor)

    # Standalone CMA-ES baseline first, so it sits behind everything.
    if baseline_trace is not None and baseline_trace.evaluations:
        ax.semilogy(
            baseline_trace.evaluations,
            clip(baseline_trace.best_fitness),
            color=baseline_color,
            linewidth=1.6,
            linestyle=":",
            alpha=0.7,
            label=f"{baseline_label}  (f = {baseline_trace.final_fitness:.2e})",
            zorder=1,
        )

    # CMA-ES backbone: best over CMA-ES generations only, ignoring the
    # L-BFGS-B drops.
    if result.cmaes_evaluations:
        ax.semilogy(
            result.cmaes_evaluations,
            clip(result.cmaes_best),
            color=cmaes_color,
            linewidth=2.0,
            alpha=0.9,
            label=cmaes_label,
            zorder=2,
        )

    # Overall-best staircase.
    ax.semilogy(
        result.overall_evaluations,
        clip(result.overall_best),
        color=overall_color,
        linewidth=3.0,
        label=f"{overall_label}  (f = {result.trace.final_fitness:.2e})",
        zorder=3,
    )

    # L-BFGS-B bursts overplotted on the drops (single legend entry).
    for index, (segment_evaluations, segment_best) in enumerate(result.burst_segments):
        ax.semilogy(
            segment_evaluations,
            clip(segment_best),
            color=burst_color,
            linewidth=2.6,
            label=burst_label if index == 0 else None,
            zorder=4,
        )

    if mark_intervals:
        for start in result.burst_starts:
            ax.axvline(
                start,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.22,
                zorder=0,
            )

    ax.set_xlabel("Function Evaluations", fontsize=12)
    ax.set_ylabel("Best Fitness (log)", fontsize=12)
    ax.set_title(
        title
        or (
            f"Interleaved CMA-ES <-> L-BFGS-B  "
            f"({result.num_bursts} bursts, {result.cmaes_generations} CMA-ES gens)"
        ),
        fontsize=13,
    )
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    fig.tight_layout()

    _save_if_path(fig, save_path)
    return fig
