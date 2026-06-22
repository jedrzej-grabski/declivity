"""Declarative diagnostic plotting — single-run entry points.

Two functions, both thin shells over the shared renderer in
:py:mod:`src.plotting.unified`:

- :py:func:`plot_metrics` — every (or selected) panel for a single run.
  Just :func:`~src.plotting.unified.plot_panels` handed one
  ``OptimizationResult``.
- :py:func:`plot_comparison` — one panel per semantic key, every algorithm
  overlaid on the same axes.

Both consume ``OptimizationResult``s and render through the same
``draw_groups`` / ``draw_single_run`` core that the benchmark plotters use,
so a :class:`~src.plotting.panel.Panel` registered once drives single runs,
overlays, and multi-seed bands alike. Adding a new panel is one line in
:py:mod:`src.plotting.standard_panels`.

(``plot_evaluation_bars`` lives here too — a non-panel summary bar chart.)
"""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.core.base_optimizer import OptimizationResult
from src.plotting.panel import PanelRegistry
from src.plotting.types import PanelKey
from src.plotting.unified import (
    PanelSelection,
    RunGroup,
    _field_for,
    _reference_panel,
    decorate_axes,
    draw_groups,
    grid_dims,
    plot_panels,
    save_if_path,
)


def plot_metrics(
    result: OptimizationResult,
    panels: PanelSelection = None,
    *,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (6.0, 4.0),
    title: str | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot diagnostic panels for a single optimization run.

    A single run is the ``len(runs) == 1`` case of the unified plotter, so
    this is :func:`~src.plotting.unified.plot_panels` over one
    ``OptimizationResult``: each panel's series are drawn as lines (the
    best/mean/median convergence overlay, σ, condition number, ...). Run the
    *same* panels over a benchmark and they come out as aggregated bands.

    Args:
        result: An optimization result. ``result.algorithm`` selects the
            registry bucket.
        panels: ``None`` (default) -> the algorithm's default panel set;
            ``"all"`` -> every registered panel; a list of panel keys
            (strings) or :class:`Panel` objects -> exact selection.
        ncols: Number of columns in the grid. Rows are inferred.
        figsize_per_panel: ``(width, height)`` in inches for each panel.
        title: Optional figure-level suptitle.
        save_path: If set, the figure is saved here (parent dir is created).

    Returns:
        The matplotlib :class:`Figure` (still open — caller may close it).
    """
    return plot_panels(
        result,
        panels,
        ncols=ncols,
        figsize_per_panel=figsize_per_panel,
        title=title,
        save_path=save_path,
    )


def plot_evaluation_bars(
    results: dict[str, OptimizationResult],
    *,
    colors: dict[str, str] | None = None,
    title: str = "Function Evaluations",
    figsize: tuple[float, float] | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """Horizontal bar chart of total evaluations per algorithm.

    Useful for runs that hit a convergence criterion at very different
    points: the convergence plot says "they all converged"; the bar
    chart says "...one of them needed 30x as many evals to get there".

    Args:
        results: ``{label: OptimizationResult}``. The bars appear in
            insertion order (top to bottom), with the label drawn at left.
        colors: Optional ``{label: hex_color}``. Missing entries fall back
            to a neutral gray.
        title: Chart title.
        figsize: Override the auto-computed figure size.
        save_path: If set, the figure is saved here.

    Returns:
        The matplotlib :class:`Figure`.
    """
    if not results:
        raise ValueError("results must contain at least one entry")

    labels = list(results.keys())
    evaluations = [results[label].evaluations for label in labels]

    if figsize is None:
        figsize = (10.0, max(3.0, len(labels) * 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    bar_colors = [
        (colors.get(label, "#888888") if colors else "#888888")
        for label in labels
    ]
    bars = ax.barh(labels, evaluations, color=bar_colors, edgecolor="white")

    max_evals = max(evaluations) if evaluations else 1
    for bar, value in zip(bars, evaluations):
        ax.text(
            bar.get_width() + max_evals * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Function Evaluations", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()  # first key at top — matches dict-insertion order
    fig.tight_layout()

    save_if_path(fig, save_path)
    return fig


def plot_comparison(
    results: dict[str, OptimizationResult],
    panels: Sequence[PanelKey | str] | None = None,
    *,
    colors: dict[str, str] | None = None,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (6.0, 4.0),
    title: str | None = None,
    save_path: Path | str | None = None,
    handoff_eval: int | None = None,
    handoff_iter: int | None = None,
) -> Figure:
    """Overlay multiple algorithms in each panel, one panel per semantic key.

    Each panel key resolves separately for each algorithm — that's the whole
    point. CMA-ES "step_size" plots ``sigma``, DES "step_size" plots ``Ft``,
    both end up on the same axes labeled by the panel's title. For
    multi-series panels, only the **first** series is used per algorithm (the
    overlay axis is the algorithm, not the field).

    Args:
        results: ``{label: OptimizationResult}``. The label appears in the
            legend and (if provided) keys into ``colors``.
        panels: A list of semantic keys. ``None`` means "intersection of keys
            registered for every algorithm in ``results``".
        colors: Optional ``{label: hex_color}`` overrides. Missing labels fall
            back to matplotlib's default cycle.
        ncols, figsize_per_panel, title, save_path: As in
            :py:func:`plot_metrics`.
        handoff_eval / handoff_iter: If set, draw a dashed vertical marker on
            the panels whose x-axis matches (evaluations vs iteration).

    Returns:
        The matplotlib :class:`Figure`.
    """
    if not results:
        raise ValueError("results must contain at least one entry")

    groups = [
        RunGroup.from_result(result, label=label, color=(colors or {}).get(label))
        for label, result in results.items()
    ]
    algorithms = [group.algorithm for group in groups if group.algorithm is not None]

    if panels is None:
        panel_keys = PanelRegistry.common(algorithms)
        if not panel_keys:
            algo_names = [a.value for a in algorithms]
            raise ValueError(
                f"No common panels across {algo_names}. "
                "Pass panels=[...] explicitly, or register more shared keys."
            )
    else:
        panel_keys = [str(key) for key in panels]

    rows, cols = grid_dims(len(panel_keys), ncols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for idx, key in enumerate(panel_keys):
        ax = flat_axes[idx]
        reference = _reference_panel(algorithms, key)
        if reference is None:
            ax.set_visible(False)
            continue

        # One line per algorithm; each draws its own field for this semantic
        # key (sigma vs Ft vs ...). Single run each, so no aggregation.
        draw_groups(
            ax,
            groups,
            field=lambda g, k=key, ref=reference: _field_for(g, k, ref),
            x_field=reference.x_field,
            floor=reference.floor,
            aggregate=False,
            default_colors=colors,
        )
        decorate_axes(ax, reference)

        # Handoff marker, routed to the panel that matches the caller's x-axis.
        if handoff_eval is not None and reference.x_field == "evaluations":
            ax.axvline(handoff_eval, color="black", linestyle="--", linewidth=1.2, alpha=0.45)
        if handoff_iter is not None and reference.x_field == "iteration":
            ax.axvline(handoff_iter, color="black", linestyle="--", linewidth=1.2, alpha=0.45)

        ax.legend(fontsize=9, framealpha=0.9)

    for idx in range(len(panel_keys), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    save_if_path(fig, save_path)
    return fig
