"""
Line-search strategies — a shared, injected framework component.

Two branches, split by what they require of the objective:

- :mod:`~declivity.utils.line_search.gradient` — searches that need the
  directional derivative ``phi'(alpha)`` (More-Thuente strong-Wolfe,
  Armijo backtracking).  Used by L-BFGS-B.
- :mod:`~declivity.utils.line_search.derivative_free` — searches that
  see only ``phi(alpha)`` (Brent with automatic bracketing, bounded
  golden-section/parabolic, pure golden section).  Used by Powell.

Like :class:`~declivity.utils.stopping_conditions.StoppingCondition` and
:class:`~declivity.utils.constraint_handlers.ConstraintHandler`, a line
search is injected into the optimizer at construction time
(``line_search=``) and defaults per algorithm (More-Thuente for
L-BFGS-B, Brent for Powell).
"""

from declivity.utils.line_search.bounds import max_feasible_step
from declivity.utils.line_search.derivative_free import (
    BracketError,
    BrentLineSearch,
    DerivativeFreeLineSearch,
    DerivativeFreeLineSearchType,
    GoldenSectionLineSearch,
    ScalarSearchResult,
    bounded_minimize,
    bracket_minimum,
    brent_minimize,
)
from declivity.utils.line_search.gradient import (
    ArmijoBacktracking,
    GradientLineSearch,
    GradientLineSearchType,
    LineSearchResult,
    LineSearchStrategy,
    MoreThuenteLineSearch,
    armijo_search,
    more_thuente_search,
)

__all__ = [
    "ArmijoBacktracking",
    "BracketError",
    "BrentLineSearch",
    # Derivative-free branch
    "DerivativeFreeLineSearch",
    "DerivativeFreeLineSearchType",
    "GoldenSectionLineSearch",
    # Gradient-based branch
    "GradientLineSearch",
    "GradientLineSearchType",
    "LineSearchResult",
    "LineSearchStrategy",
    "MoreThuenteLineSearch",
    "ScalarSearchResult",
    "armijo_search",
    "bounded_minimize",
    "bracket_minimum",
    "brent_minimize",
    # Box-bound geometry (feeds stpmax for bound-constrained line searches)
    "max_feasible_step",
    "more_thuente_search",
]
