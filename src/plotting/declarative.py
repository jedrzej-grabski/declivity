"""Declarative diagnostic plotting.

Two entry points:

- :py:func:`plot_metrics` — every (or selected) panel for a single run.
- :py:func:`plot_comparison` — one panel per semantic key, every algorithm
  overlaid on the same axes.

Adding a new panel is one line in :py:mod:`src.plotting.standard_panels`
(or anywhere, by calling ``PanelRegistry.register``); both functions pick
it up automatically. ``plot_comparison(results)`` with no explicit panel
list defaults to the intersection across the algorithms in ``results``,
so the common-metric workflow is the path of least resistance.

Panels may be single-series or multi-series (best/mean/median together).
:py:func:`plot_metrics` honours all series; :py:func:`plot_comparison`
only uses each panel's first series, since its overlay axis is the
algorithm.
"""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.algorithms.choices import AlgorithmChoice
from src.core.base_optimizer import OptimizationResult
from src.logging.base_logger import BaseLogData
from src.plotting.panel import Panel, PanelRegistry
from src.plotting.types import PanelKey, PanelSet


# Accepted shapes for the ``panels=`` argument across the public API.
PanelSelection = Sequence[PanelKey | str | Panel] | PanelSet | str | None


def _resolve_panels(
    algorithm: AlgorithmChoice,
    panels: PanelSelection,
) -> list[Panel]:
    """Coerce a panel selection into concrete :class:`Panel` objects.

    Selection rules:

    - ``None``                     -> panels marked ``default=True`` for this algorithm.
    - ``PanelSet.DEFAULT``         -> same as ``None``.
    - ``PanelSet.ALL`` / ``"all"`` -> every panel registered for this algorithm.
    - ``Sequence[...]``            -> exact list. Each element may be a
                                      :class:`PanelKey`, a raw string key,
                                      or a :class:`Panel` instance (the latter
                                      passes through without a registry lookup).
    """
    if panels is None or panels is PanelSet.DEFAULT:
        keys = PanelRegistry.default(algorithm)
        return [PanelRegistry.get(algorithm, key) for key in keys]
    if isinstance(panels, (str, PanelSet)):
        # Single-value sentinel. PanelSet members are also strings (StrEnum),
        # so the equality check handles both ``"all"`` and ``PanelSet.ALL``.
        if str(panels) != str(PanelSet.ALL):
            raise ValueError(
                f"panels={panels!r} not understood. Pass a list, None, "
                f"or PanelSet.ALL."
            )
        keys = PanelRegistry.available(algorithm)
        return [PanelRegistry.get(algorithm, key) for key in keys]
    return [
        panel if isinstance(panel, Panel) else PanelRegistry.get(algorithm, panel)
        for panel in panels
    ]


def _grid_dims(num_panels: int, ncols: int) -> tuple[int, int]:
    """Rows / cols for ``num_panels`` cells, respecting ``ncols``."""
    cols = max(1, min(ncols, num_panels))
    rows = (num_panels + cols - 1) // cols
    return rows, cols


def _extract_series(log_data: BaseLogData, field: str) -> list:
    """Return ``log_data.<field>`` as a list, or ``[]`` if missing/empty."""
    return list(getattr(log_data, field, None) or [])


def _apply_floor(values: list[float], floor: float | None) -> list[float]:
    """Clip below ``floor`` if set (keeps log-scale plots finite)."""
    if floor is None:
        return values
    return [value if value > floor else floor for value in values]


def _draw_one_series(
    ax: Axes,
    x_values: list,
    y_values: list,
    *,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> bool:
    """Draw one line. Returns ``False`` if the series was empty."""
    if not x_values or not y_values:
        return False

    # Length mismatches happen when a metric is only logged when a diag
    # flag is set (e.g. condition_number only when diag_eigen=True).
    # Truncate to the common prefix so the plot still draws something
    # honest.
    length = min(len(x_values), len(y_values))
    ax.plot(
        x_values[:length],
        y_values[:length],
        label=label,
        color=color,
        linestyle=str(linestyle),  # LineStyle (StrEnum) or raw string
        linewidth=linewidth,
    )
    return True


def _draw_panel(
    ax: Axes,
    log_data: BaseLogData,
    panel: Panel,
    *,
    legend_for_series: bool,
) -> None:
    """Render every :class:`Series` of one :class:`Panel` onto ``ax``.

    ``legend_for_series`` controls whether the in-panel legend is drawn
    (only useful for multi-series single-run plots, not comparison overlays
    where the legend is per-algorithm at the panel level).
    """
    x_values = _extract_series(log_data, panel.x_field)
    drew_anything = False
    for series in panel.series_list:
        y_values = _apply_floor(
            _extract_series(log_data, series.field),
            panel.floor,
        )
        label = series.display_label if legend_for_series else None
        if _draw_one_series(
            ax,
            x_values,
            y_values,
            label=label,
            color=series.color,
            linestyle=series.linestyle,
        ):
            drew_anything = True

    _decorate(ax, panel)
    if drew_anything and legend_for_series and len(panel.series_list) > 1:
        ax.legend(fontsize=9, framealpha=0.9)


def _decorate(ax: Axes, panel: Panel) -> None:
    """Apply title, labels, scale, grid to an axis from a Panel spec."""
    ax.set_xlabel(panel.x_field.replace("_", " ").title())
    ax.set_ylabel(panel.ylabel)
    ax.set_title(panel.title)
    ax.set_yscale(str(panel.yscale))  # YScale (StrEnum) or raw string
    ax.grid(True, alpha=0.3, which="both")


def _save_if_path(fig: Figure, save_path: Path | str | None) -> None:
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


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
    resolved = _resolve_panels(result.algorithm, panels)
    if not resolved:
        raise ValueError(
            f"No panels to plot for {result.algorithm}. "
            f"Register some via PanelRegistry.register, mark some as default=True, "
            f"or pass panels=... explicitly."
        )

    rows, cols = _grid_dims(len(resolved), ncols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    log_data = result.diagnostic
    for idx, panel in enumerate(resolved):
        _draw_panel(flat_axes[idx], log_data, panel, legend_for_series=True)

    for idx in range(len(resolved), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    _save_if_path(fig, save_path)
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
) -> Figure:
    """Overlay multiple algorithms in each panel, one panel per semantic key.

    Each panel key resolves separately for each algorithm — that's the
    whole point. CMA-ES "step_size" plots ``sigma``, DES "step_size" plots
    ``Ft``, both end up on the same axes labeled by the panel's title.

    For multi-series panels, only the **first** series is used in each
    algorithm's curve (the overlay axis is the algorithm, not the field).

    Args:
        results: ``{label: OptimizationResult}``. The label appears in the
            legend and (if provided) keys into ``colors``.
        panels: A list of semantic keys. ``None`` means "intersection of
            keys registered for every algorithm in ``results``".
        colors: Optional ``{label: hex_color}`` overrides. Missing labels
            fall back to matplotlib's default cycle.
        ncols, figsize_per_panel, title, save_path: Same as
            :py:func:`plot_metrics`.

    Returns:
        The matplotlib :class:`Figure`.
    """
    if not results:
        raise ValueError("results must contain at least one entry")

    algorithms = [result.algorithm for result in results.values()]

    if panels is None:
        panel_keys = PanelRegistry.common(algorithms)
        if not panel_keys:
            algo_names = [a.value for a in algorithms]
            raise ValueError(
                f"No common panels across {algo_names}. "
                "Pass panels=[...] explicitly, or register more shared keys."
            )
    else:
        # Normalize to plain str so PanelKey enums and raw strings both work.
        panel_keys = [str(key) for key in panels]

    rows, cols = _grid_dims(len(panel_keys), ncols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for idx, key in enumerate(panel_keys):
        ax = flat_axes[idx]

        # Reference panel: title / ylabel / scale come from the first
        # algorithm. They're semantic matches by registration, so any of
        # them would do.
        reference_panel = PanelRegistry.get(algorithms[0], key)

        for label, result in results.items():
            try:
                panel = PanelRegistry.get(result.algorithm, key)
            except KeyError:
                # Algorithm didn't register this key. Only possible when
                # the caller passed panels=... with a key that isn't
                # registered everywhere; skip silently and move on.
                continue
            # Only the panel's primary series participates in the overlay.
            primary_series = panel.series_list[0]
            x_values = _extract_series(result.diagnostic, panel.x_field)
            y_values = _apply_floor(
                _extract_series(result.diagnostic, primary_series.field),
                panel.floor,
            )
            color = colors.get(label) if colors else None
            _draw_one_series(
                ax,
                x_values,
                y_values,
                label=label,
                color=color,
            )

        _decorate(ax, reference_panel)
        ax.legend(fontsize=9, framealpha=0.9)

    for idx in range(len(panel_keys), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    _save_if_path(fig, save_path)
    return fig
