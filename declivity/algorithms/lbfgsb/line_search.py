"""
Compatibility shim — the line-search implementations moved to
:mod:`declivity.utils.line_search` when the line search became a shared
framework component rather than an L-BFGS-B internal.

Import from ``declivity.utils.line_search`` in new code.
"""

from declivity.utils.line_search.gradient import (
    ArmijoBacktracking,
    GradientLineSearch,
    LineSearchResult,
    LineSearchStrategy,
    MoreThuenteLineSearch,
    armijo_search,
    more_thuente_search,
)

__all__ = [
    "LineSearchResult",
    "GradientLineSearch",
    "LineSearchStrategy",
    "MoreThuenteLineSearch",
    "ArmijoBacktracking",
    "more_thuente_search",
    "armijo_search",
]
