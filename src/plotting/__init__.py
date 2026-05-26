"""Plotting package.

Two entry points for diagnostic plots:

- :py:func:`plot_metrics` — every (or selected) panel for a single run.
- :py:func:`plot_comparison` — semantic-key panels overlaid across runs.

Adding a new panel: one ``PanelRegistry.register(...)`` call. See
:py:mod:`src.plotting.standard_panels` for the four built-in algorithms.

The legacy :class:`MultiAlgorithmPlotter` is still re-exported so existing
experiments keep working during migration.
"""

# Importing standard_panels triggers all panel registrations as a side
# effect. Keep this before re-exports so downstream imports see a populated
# registry.
from src.plotting import standard_panels  # noqa: F401

from src.plotting.benchmark import (
    plot_benchmark_boxplot,
    plot_benchmark_convergence,
)
from src.plotting.declarative import plot_comparison, plot_metrics
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter
from src.plotting.panel import Panel, PanelRegistry

__all__ = [
    "MultiAlgorithmPlotter",
    "Panel",
    "PanelRegistry",
    "plot_benchmark_boxplot",
    "plot_benchmark_convergence",
    "plot_comparison",
    "plot_metrics",
]
