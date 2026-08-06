"""
Compatibility shim — the line-search implementations moved to
:mod:`declivity.utils.line_search` when the line search became a shared
framework component (gradient-based branch used by L-BFGS-B,
derivative-free branch used by Powell).

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
    "ArmijoBacktracking",
    "GradientLineSearch",
    "LineSearchResult",
    "LineSearchStrategy",
    "MoreThuenteLineSearch",
    "armijo_search",
    "more_thuente_search",
]
