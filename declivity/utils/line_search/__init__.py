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

from declivity.utils.line_search.gradient import (
    ArmijoBacktracking,
    GradientLineSearch,
    LineSearchResult,
    LineSearchStrategy,
    MoreThuenteLineSearch,
    armijo_search,
    more_thuente_search,
)
from declivity.utils.line_search.derivative_free import (
    BracketError,
    BrentLineSearch,
    DerivativeFreeLineSearch,
    GoldenSectionLineSearch,
    ScalarSearchResult,
    bounded_minimize,
    bracket_minimum,
    brent_minimize,
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
    # Derivative-free branch
    "DerivativeFreeLineSearch",
    "ScalarSearchResult",
    "BrentLineSearch",
    "GoldenSectionLineSearch",
    "BracketError",
    "bracket_minimum",
    "brent_minimize",
    "bounded_minimize",
]
