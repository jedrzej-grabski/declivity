"""Plotting package.

Entry points for diagnostic plots:

- :py:func:`plot_metrics` — every (or selected) panel for a single run.
- :py:func:`plot_comparison` — semantic-key panels overlaid across runs.
- :py:func:`plot_evaluation_bars` — horizontal bar chart of total
  evaluations per labeled run.
- :py:func:`plot_benchmark_convergence` — multi-seed convergence grid
  with median + IQR band per algorithm.
- :py:func:`plot_benchmark_boxplot` — multi-seed final-fitness boxplot.
- :py:func:`plot_function_landscape` /
  :py:func:`plot_function_landscape_grid` — 2D contour plots of an
  objective function.
- :py:func:`plot_matrix_diagonal_comparison` — specialized sorted-diagonal
  comparison against a reference matrix.

Adding a new panel: one ``PanelRegistry.register(...)`` call. See
:py:mod:`src.plotting.standard_panels` for the four built-in algorithms.
"""

# Importing standard_panels triggers all panel registrations as a side
# effect. Keep this before re-exports so downstream imports see a populated
# registry.
from src.plotting import standard_panels  # noqa: F401

from src.plotting.benchmark import (
    plot_benchmark_boxplot,
    plot_benchmark_convergence,
    plot_convergence_overlay,
)
from src.plotting.declarative import (
    plot_comparison,
    plot_evaluation_bars,
    plot_metrics,
)
from src.plotting.diagnostics import plot_matrix_diagonal_comparison
from src.plotting.landscape import (
    plot_function_landscape,
    plot_function_landscape_grid,
)
from src.plotting.panel import Panel, PanelRegistry, Series
from src.plotting.types import LineStyle, PanelKey, PanelSet, XAxis, YScale
from src.plotting.unified import RunGroup, RunSeries, draw_groups, plot_panels

__all__ = [
    "LineStyle",
    "Panel",
    "PanelKey",
    "PanelRegistry",
    "PanelSet",
    "RunGroup",
    "RunSeries",
    "Series",
    "XAxis",
    "YScale",
    "draw_groups",
    "plot_benchmark_boxplot",
    "plot_benchmark_convergence",
    "plot_comparison",
    "plot_convergence_overlay",
    "plot_evaluation_bars",
    "plot_function_landscape",
    "plot_function_landscape_grid",
    "plot_matrix_diagonal_comparison",
    "plot_metrics",
    "plot_panels",
]
