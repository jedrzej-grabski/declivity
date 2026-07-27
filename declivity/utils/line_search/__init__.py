"""
Line-search strategies — a shared, injected framework component.

:mod:`~declivity.utils.line_search.gradient` holds the searches that need
the directional derivative ``phi'(alpha)`` (More-Thuente strong-Wolfe,
Armijo backtracking).  Used by L-BFGS-B.

Like :class:`~declivity.utils.stopping_conditions.StoppingCondition` and
:class:`~declivity.utils.constraint_handlers.ConstraintHandler`, a line
search is injected into the optimizer at construction time
(``line_search=``) and defaults per algorithm (More-Thuente for
L-BFGS-B).
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
    # Gradient-based branch
    "GradientLineSearch",
    "LineSearchStrategy",
    "LineSearchResult",
    "MoreThuenteLineSearch",
    "ArmijoBacktracking",
    "more_thuente_search",
    "armijo_search",
]
