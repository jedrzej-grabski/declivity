"""The unified panel plotter — one renderer for single runs *and* benchmarks.

The framework used to draw single-run diagnostics (``plot_metrics`` over an
``OptimizationResult``) and multi-seed benchmark curves (``plot_benchmark_*``
over ``RunTrace`` lists) on two unrelated code paths. They are now the same
thing at different seed counts:

- **a run** exposes named per-step series via ``get_series(field)`` — both a
  live ``LogData`` (single run, every field) and a persisted ``RunTrace``
  (trimmed to the retained scalar fields) satisfy this :class:`RunSeries`
  contract;
- **a result** is a :class:`RunGroup` — 1..N runs of one algorithm. N=1 is a
  single run (drawn as a line); N=25 is a benchmark (drawn as a median + IQR
  band).

:func:`draw_groups` is the shared atomic renderer: it draws one line-or-band
per group for a single field, and *that one function* is what every public
entry point (``plot_metrics``, ``plot_comparison``, ``plot_benchmark_convergence``,
``plot_convergence_overlay``, and the new :func:`plot_panels`) renders through.
So a :class:`~declivity.plotting.panel.Panel` registered once drives all of them.

:func:`plot_panels` is the unified front door: pass it a single
``OptimizationResult`` and it draws that run's panels as lines; pass it a
benchmark (a :class:`RunGroup` of many seeds, or a dict of them) and the same
panels come out as aggregated bands. Same panel, same call.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from declivity.algorithms.choices import AlgorithmChoice
from declivity.benchmarking.aggregation import (
    percentile_band,
    series_grid,
    stack_runs_on_grid,
)
from declivity.core.base_optimizer import OptimizationResult
from declivity.plotting.panel import Panel, PanelRegistry
from declivity.plotting.types import PanelKey, PanelSet

# Accepted shapes for the ``panels=`` argument across the public API. Defined
# here (not in types.py) because it references Panel, which would cycle.
PanelSelection = Sequence["PanelKey | str | Panel"] | PanelSet | str | None


# ===========================================================================
# The unified data model
# ===========================================================================


@runtime_checkable
class RunSeries(Protocol):
    """A single run's named per-step arrays.

    Satisfied by both :class:`~declivity.logging.base_logger.BaseLogData` (a live
    single run, every field) and :class:`~declivity.benchmarking.run_trace.RunTrace`
    (a trimmed, persisted run). ``get_series`` returns the array for a panel's
    ``field`` / ``x_field`` by name, or ``None`` when that field wasn't
    retained — which the plotter renders as an empty series, never an error.
    """

    def get_series(self, name: str) -> list[float] | None: ...


@dataclass
class RunGroup:
    """1..N runs of one algorithm on one problem — the unit the plotter draws.

    A single optimization is a group of one run (→ a line); a benchmark is a
    group of many seeds (→ a median + IQR band). Heterogeneous groups overlay
    on one axes (the cross-algorithm comparison).
    """

    label: str
    """Legend label."""

    runs: list[RunSeries]
    """One per seed. ``len == 1`` → line; ``len > 1`` → aggregated band."""

    color: str | None = None
    """Line / band colour. ``None`` falls back to a caller-supplied palette
    or the matplotlib cycle."""

    algorithm: AlgorithmChoice | None = None
    """Which :class:`~declivity.plotting.panel.PanelRegistry` bucket this group's
    panels resolve against. Enables semantic keys across algorithms (the same
    ``step_size`` key → ``sigma`` here, ``Ft`` there). ``None`` falls back to
    a reference panel's field."""

    @classmethod
    def from_result(
        cls,
        result: OptimizationResult,
        *,
        label: str | None = None,
        color: str | None = None,
    ) -> "RunGroup":
        """Wrap a single ``OptimizationResult`` (its ``LogData`` is the one run)."""
        return cls(
            label=label or str(getattr(result.algorithm, "value", result.algorithm)),
            runs=[result.diagnostic],
            color=color,
            algorithm=result.algorithm,
        )

    @classmethod
    def from_runs(
        cls,
        label: str,
        runs: Iterable[RunSeries],
        *,
        color: str | None = None,
        algorithm: AlgorithmChoice | None = None,
    ) -> "RunGroup":
        """Wrap several runs (e.g. a benchmark's per-seed ``RunTrace`` list)."""
        return cls(label=label, runs=list(runs), color=color, algorithm=algorithm)


# ===========================================================================
# Low-level rendering primitives (shared by every plotter)
# ===========================================================================


def save_if_path(fig: Figure, save_path: Path | str | None) -> None:
    """Write ``fig`` to ``save_path`` (dpi 150), creating parent dirs."""
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


def grid_dims(num_panels: int, ncols: int) -> tuple[int, int]:
    """Rows / cols for ``num_panels`` cells, respecting ``ncols``."""
    cols = max(1, min(ncols, num_panels))
    rows = (num_panels + cols - 1) // cols
    return rows, cols


def apply_floor(values: list[float] | None, floor: float | None) -> list[float]:
    """Clip values below ``floor`` (keeps log-scale plots finite)."""
    if values is None:
        return []
    if floor is None:
        return list(values)
    return [max(floor, value) for value in values]


def draw_line(
    ax: Axes,
    x_values: list[float] | None,
    y_values: list[float] | None,
    *,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float | None = None,
) -> bool:
    """Draw one line. Returns ``False`` (drawing nothing) if either array is empty.

    Length mismatches (a field logged at a different cadence) are truncated to
    the common prefix so the plot still draws something honest.
    """
    if not x_values or not y_values:
        return False
    length = min(len(x_values), len(y_values))
    ax.plot(
        x_values[:length],
        y_values[:length],
        label=label,
        color=color,
        linestyle=str(linestyle),
        linewidth=linewidth,
        alpha=alpha,
    )
    return True


def decorate_axes(
    ax: Axes,
    panel: Panel,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid_alpha: float = 0.3,
) -> None:
    """Apply title, labels, scale, grid to an axes from a :class:`Panel`."""
    ax.set_xlabel(
        xlabel if xlabel is not None else panel.x_field.replace("_", " ").title()
    )
    ax.set_ylabel(ylabel if ylabel is not None else panel.ylabel)
    ax.set_title(title if title is not None else panel.title)
    ax.set_yscale(str(panel.yscale))
    ax.grid(True, alpha=grid_alpha, which="both")


# ===========================================================================
# The shared atomic renderers
# ===========================================================================


def _final_value(run: RunSeries, field_name: str) -> float | None:
    """A run's final value for ``field_name`` (prefers ``final_fitness`` when present)."""
    if field_name == "best_fitness":
        final = getattr(run, "final_fitness", None)
        if final is not None:
            return float(final)
    series = run.get_series(field_name)
    return float(series[-1]) if series else None


def _final_label(
    label: str,
    runs: list[RunSeries],
    field_name: str,
    annotate: str | None,
) -> str:
    """Optionally append the (median) final value to a legend label."""
    if not annotate:
        return label
    finals = [v for run in runs if (v := _final_value(run, field_name)) is not None]
    if not finals:
        return label
    if len(finals) > 1:
        return f"{label}  (median = {float(np.median(finals)):.2e})"
    return f"{label}  (f = {finals[0]:.2e})"


def draw_groups(
    ax: Axes,
    groups: Iterable[RunGroup],
    *,
    field: str | Callable[[RunGroup], str | None],
    x_field: str = "evaluations",
    floor: float | None = None,
    aggregate: bool = True,
    show_band: bool = True,
    iqr_alpha: float = 0.15,
    linewidth: float = 2.2,
    num_grid_points: int = 200,
    annotate_final: str | None = None,
    default_colors: dict[str, str] | None = None,
) -> bool:
    """Draw one line-or-band per group for a single field — the unification point.

    For each group: a single run is drawn as a raw line; multiple runs are
    aggregated into a median curve with a 25/75 IQR band (when
    ``aggregate``). ``field`` is either a fixed field name (homogeneous, e.g.
    ``best_fitness`` across algorithms) or a callable ``group -> field`` (the
    semantic-key case, where the field differs per algorithm). Returns whether
    anything was drawn.
    """
    drew = False
    for group in groups:
        field_name = field(group) if callable(field) else field
        if not field_name or not group.runs:
            continue
        color = group.color or (default_colors or {}).get(group.label)
        runs = group.runs
        label = _final_label(group.label, runs, field_name, annotate_final)

        if len(runs) == 1 or not aggregate:
            run = runs[0]
            if draw_line(
                ax,
                run.get_series(x_field),
                apply_floor(run.get_series(field_name), floor),
                label=label,
                color=color,
                linewidth=linewidth,
                alpha=0.95,
            ):
                drew = True
            continue

        grid = series_grid(runs, x_field, num_points=num_grid_points)
        matrix = stack_runs_on_grid(runs, grid, field_name, x_field)
        if floor is not None:
            matrix = np.maximum(matrix, floor)
        finite = np.where(np.isinf(matrix), np.nan, matrix)
        if finite.size == 0 or bool(np.all(np.isnan(finite))):
            continue  # no run retained this field — skip rather than draw all-NaN
        median, low, high = percentile_band(matrix)
        if draw_line(
            ax,
            grid.tolist(),
            median.tolist(),
            label=label,
            color=color,
            linewidth=linewidth,
            alpha=0.95,
        ):
            drew = True
        if show_band and matrix.shape[0] > 1:
            ax.fill_between(grid, low, high, color=color, alpha=iqr_alpha)
    return drew


def draw_single_run(
    ax: Axes,
    run: RunSeries,
    panel: Panel,
    *,
    floor: float | None = None,
) -> bool:
    """Draw every :class:`~declivity.plotting.panel.Series` of one panel from one run.

    The multi-series single-run view (best/mean/median on one axes). Distinct
    from :func:`draw_groups` (one curve per *group*); here the several curves
    come from one run's several fields.
    """
    floor_value = floor if floor is not None else panel.floor
    x_values = run.get_series(panel.x_field)
    drew = False
    for series in panel.series_list:
        if draw_line(
            ax,
            x_values,
            apply_floor(run.get_series(series.field), floor_value),
            label=series.display_label,
            color=series.color,
            linestyle=series.linestyle,
        ):
            drew = True
    return drew


# ===========================================================================
# Panel resolution + data coercion for the unified front door
# ===========================================================================


def _coerce_to_groups(
    data: object,
    colors: dict[str, str] | None,
) -> list[RunGroup]:
    """Normalise any accepted input into a list of :class:`RunGroup`."""
    if isinstance(data, OptimizationResult):
        return [RunGroup.from_result(data)]
    if isinstance(data, RunGroup):
        return [data]
    if isinstance(data, dict):
        groups: list[RunGroup] = []
        for label, value in data.items():
            label = str(label)
            color = (colors or {}).get(label)
            if isinstance(value, OptimizationResult):
                groups.append(RunGroup.from_result(value, label=label, color=color))
            elif isinstance(value, RunGroup):
                if value.color is None:
                    value.color = color
                if not value.label:
                    value.label = label
                groups.append(value)
            elif isinstance(value, (list, tuple)):
                groups.append(RunGroup.from_runs(label, value, color=color))
            else:
                raise TypeError(
                    f"plot_panels: dict value for {label!r} must be an "
                    f"OptimizationResult, a RunGroup, or a list of runs."
                )
        return groups
    if isinstance(data, (list, tuple)):
        if all(isinstance(item, RunGroup) for item in data):
            return list(data)
        raise TypeError("plot_panels: a list input must contain RunGroup objects.")
    raise TypeError(
        f"plot_panels: unsupported data type {type(data).__name__}. Pass an "
        f"OptimizationResult, a dict of results, or RunGroup(s)."
    )


def _reference_panel(algos: list[AlgorithmChoice], key: str) -> Panel | None:
    """First registered panel for ``key`` across ``algos`` (title/scale source)."""
    for algo in algos:
        try:
            return PanelRegistry.get(algo, key)
        except KeyError:
            continue
    return None


def _resolve_panel_list(
    panels: PanelSelection,
    algos: list[AlgorithmChoice],
    metrics_mode: bool,
) -> list[tuple[str, Panel]]:
    """Resolve a panel selection into ``(key, reference_panel)`` pairs.

    ``None`` → the single algorithm's defaults (one group) or the intersection
    across algorithms (overlay). ``"all"`` → every / common registered panel.
    A list may hold panel keys or :class:`Panel` objects (the latter pass
    through, so a study can plot an ad-hoc panel without registering it).
    """
    # Explicit Panel objects pass straight through.
    if (
        isinstance(panels, (list, tuple))
        and panels
        and all(isinstance(p, Panel) for p in panels)
    ):
        return [(str(p.key), p) for p in panels]  # type: ignore[union-attr]

    unique_algos = list(dict.fromkeys(algos))
    if panels is None or panels is PanelSet.DEFAULT:
        if not unique_algos:
            return []
        if metrics_mode or len(unique_algos) == 1:
            keys: list[str] = PanelRegistry.default(unique_algos[0])
        else:
            keys = PanelRegistry.common(unique_algos)
    elif isinstance(panels, (str, PanelSet)) and str(panels) == str(PanelSet.ALL):
        if not unique_algos:
            return []
        keys = (
            PanelRegistry.available(unique_algos[0])
            if (metrics_mode or len(unique_algos) == 1)
            else PanelRegistry.common(unique_algos)
        )
    else:
        keys = [str(key) for key in panels]  # type: ignore[union-attr]

    resolved: list[tuple[str, Panel]] = []
    for key in keys:
        reference = _reference_panel(unique_algos, key)
        if reference is not None:
            resolved.append((str(key), reference))
    return resolved


def _field_for(group: RunGroup, key: str, reference: Panel) -> str | None:
    """The field this group plots for panel ``key`` (semantic resolution)."""
    if group.algorithm is not None:
        try:
            return PanelRegistry.get(group.algorithm, key).series_list[0].field
        except KeyError:
            pass
    return reference.series_list[0].field


# ===========================================================================
# The unified front door
# ===========================================================================


def plot_panels(
    data: object,
    panels: PanelSelection = None,
    *,
    colors: dict[str, str] | None = None,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (6.0, 4.0),
    title: str | None = None,
    aggregate: bool = True,
    show_band: bool = True,
    annotate_final: str | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot panels from a single run *or* a benchmark — one call, one panel set.

    Args:
        data: One of —
            * an ``OptimizationResult`` (single run → lines);
            * a ``dict[label, OptimizationResult]`` (overlay of single runs);
            * a :class:`RunGroup` or list/dict of them (a benchmark of many
              seeds → median + IQR bands; ``len(runs) == 1`` falls back to a
              line). Build one with ``RunGroup.from_runs(name, traces,
              color=..., algorithm=...)``.
        panels: ``None`` → the algorithm's default panels (one source) or the
            intersection across algorithms (overlay); ``"all"`` /
            ``PanelSet.ALL`` → all registered; a list of keys / ``Panel``
            objects → exact selection.
        colors: ``{label: hex}`` fallback when a group has no ``color``.
        aggregate: When a group has many runs, draw median + band (``True``)
            or just the first run (``False``).
        annotate_final: ``"median"`` / ``"final"`` to append the final value to
            each legend label; ``None`` (default) to omit.

    Returns:
        The matplotlib :class:`Figure`.
    """
    groups = _coerce_to_groups(data, colors)
    if not groups:
        raise ValueError("plot_panels: no data to plot.")

    metrics_mode = len(groups) == 1 and len(groups[0].runs) == 1
    algos = [g.algorithm for g in groups if g.algorithm is not None]

    resolved = _resolve_panel_list(panels, algos, metrics_mode)
    if not resolved:
        raise ValueError(
            "plot_panels: no panels to render. Pass panels=[...] explicitly "
            "(no algorithm metadata was available to resolve defaults)."
        )

    rows, cols = grid_dims(len(resolved), ncols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for idx, (key, reference) in enumerate(resolved):
        ax = flat_axes[idx]
        if metrics_mode:
            drew = draw_single_run(ax, groups[0].runs[0], reference)
            decorate_axes(ax, reference)
            if drew and len(reference.series_list) > 1:
                ax.legend(fontsize=9, framealpha=0.9)
        else:
            drew = draw_groups(
                ax,
                groups,
                field=lambda g, k=key, ref=reference: _field_for(g, k, ref),
                x_field=reference.x_field,
                floor=reference.floor,
                aggregate=aggregate,
                show_band=show_band,
                annotate_final=annotate_final,
                default_colors=colors,
            )
            decorate_axes(ax, reference)
            if drew:
                ax.legend(fontsize=9, framealpha=0.9, loc="best")

    for idx in range(len(resolved), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    save_if_path(fig, save_path)
    return fig
