"""
Box-bound geometry helpers shared by line-search-driven optimizers.

:func:`max_feasible_step` is the directional ratio-test that turns a pair
of per-dimension bound arrays into the scalar ``stpmax`` a
:class:`~declivity.utils.line_search.gradient.GradientLineSearch` expects —
the largest ``alpha >= 0`` such that ``x + alpha * direction`` stays inside
``[lower_bounds, upper_bounds]``.

This is deliberately separate from
:mod:`declivity.utils.constraint_handlers`: a ``ConstraintHandler`` answers
"is this *point* feasible, and how do I repair it if not"; this module
answers "how far can I move *along this direction* before I'd leave the
box." Algorithm-specific policy (e.g. L-BFGS-B's convention of always
trying ``alpha = 1`` on the first iteration) belongs at the call site, not
here.
"""

import numpy as np
from numpy.typing import NDArray


def max_feasible_step(
    x: NDArray[np.float64],
    direction: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> float:
    """Largest ``alpha >= 0`` such that ``x + alpha * direction`` stays in the box.

    Parameters
    ----------
    x:
        Current point (assumed feasible).
    direction:
        Search direction.
    lower_bounds, upper_bounds:
        Per-dimension bound arrays.

    Returns
    -------
    float
        ``0.0`` if any component of *direction* points straight at or past
        a bound already touched, otherwise the smallest per-dimension
        distance-to-bound ratio (capped at ``1e10``).
    """
    # Exact per-component step to the bound faced by each direction
    # component; zero components never limit the step (-> inf).  Taking
    # the exact minimum guarantees x + alpha*d stays inside the box, so
    # a belt-and-braces clip after the line search is a no-op and cached
    # f/gradient values always describe the accepted point.
    with np.errstate(divide="ignore", invalid="ignore"):
        steps_to_bound = np.where(
            direction > 0,
            (upper_bounds - x) / direction,
            np.where(
                direction < 0,
                (lower_bounds - x) / direction,
                np.inf,
            ),
        )

    max_step = float(min(np.min(steps_to_bound), 1e10))
    return max(max_step, 0.0)
