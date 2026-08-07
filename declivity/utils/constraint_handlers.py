"""
Constraint handler abstractions and box-constraint implementations.

A :class:`ConstraintHandler` defines the feasible region for a run.
Optimizers, population initializers, and gradient strategies all query it
rather than reading bound arrays directly.

Only the feasibility tests are abstract.  The geometry methods
(``feasible_step_interval``, ``max_feasible_step``, ``project_direction``,
``projected_gradient``) default to the answers implied by ``bounding_box``,
which itself defaults to unbounded.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from declivity.utils.optimality import projected_gradient as _box_projected_gradient

MAX_FEASIBLE_STEP = 1e10
"""Finite cap returned by :meth:`ConstraintHandler.max_feasible_step` for a ray
that is unbounded in the search direction."""


def _box_step_interval(
    x: NDArray[np.float64],
    direction: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> tuple[float, float]:
    """Ratio test: the span of ``x + alpha * direction`` that stays in a box.

    For each coordinate the direction actually moves, the two bounds give the
    step lengths at which that coordinate reaches them; the feasible interval
    is the intersection over coordinates.  Coordinates with a zero direction
    component are excluded.

    Either bound may be infinite, in which case the matching end of the
    interval is infinite too.  A returned ``(0.0, 0.0)`` means no non-zero
    step is feasible.
    """
    (nonzero,) = np.asarray(direction).nonzero()
    if nonzero.size == 0:
        return (-np.inf, np.inf)

    lower = lower_bounds[nonzero]
    upper = upper_bounds[nonzero]
    x_nz = x[nonzero]
    d_nz = direction[nonzero]
    low = (lower - x_nz) / d_nz
    high = (upper - x_nz) / d_nz

    # Moving in +d hits the upper bound last, in -d the lower bound.
    pos = d_nz > 0
    alpha_min = float(np.max(np.where(pos, low, 0) + np.where(pos, 0, high)))
    alpha_max = float(np.min(np.where(pos, high, 0) + np.where(pos, 0, low)))

    return (alpha_min, alpha_max) if alpha_max >= alpha_min else (0.0, 0.0)


class BoxStrategy(Enum):
    """Repair strategy for a box-constrained domain."""

    CLAMP = "clamp"
    BOUNCE_BACK = "bounce_back"


class ConstraintHandler(ABC):
    """
    Abstract base class for constraint-handling strategies.

    A ConstraintHandler covers five concerns:

    * **Feasibility test** — ``is_feasible`` / ``feasibility_distance``
      (abstract).
    * **Repair** — project or bounce an infeasible point back into the
      feasible region.  Default is a no-op.
    * **Penalty** — augment an objective value to discourage infeasibility.
      Default is a no-op.
    * **Enclosing box** — ``bounding_box``, the tightest axis-aligned box
      containing the feasible region.  Defaults to unbounded.
    * **Directional feasibility** — ``feasible_step_interval`` /
      ``max_feasible_step`` / ``project_direction`` / ``projected_gradient``,
      all defaulting to the answers implied by ``bounding_box``.
    """

    @abstractmethod
    def is_feasible(self, x: NDArray[np.float64]) -> bool:
        """Return True iff *x* satisfies all constraints."""
        ...

    @abstractmethod
    def feasibility_distance(self, x: NDArray[np.float64]) -> float:
        """
        Return a non-negative scalar measuring how far *x* is from the
        feasible region (0.0 if feasible).

        The exact metric is implementation-defined but must be non-negative
        and zero if and only if *x* is feasible.
        """
        ...

    def repair(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Return a (possibly modified) copy of *x* that is closer to or inside
        the feasible region.

        The default implementation returns *x* unchanged.
        """
        return x

    def repair_batch(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Repair every row of *population* and return the result.

        Default implementation loops per row calling :meth:`repair`.
        Subclasses with a vectorised projection should override this for
        performance — boxes, ellipsoids, etc. can usually do the whole
        matrix in one call.
        """
        return np.array([self.repair(row) for row in population])

    def penalty(self, x: NDArray[np.float64], f_x: float) -> float:
        """
        Return an augmented objective value for *x* with objective *f_x*.

        The default implementation returns *f_x* unchanged (no penalty).
        """
        return f_x

    # Geometry — everything below defaults to a value derived from
    # bounding_box.

    def bounding_box(
        self, dimensions: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Tightest axis-aligned box ``(lower, upper)`` containing the feasible
        region, as two arrays of length *dimensions*.

        Coordinate-wise algorithms (L-BFGS-B's Cauchy point, population
        sampling, finite-difference probes) read their bounds from here.  The
        default is unbounded.  A handler with a bounded feasible set should
        override this even when that set is not a box; a conservative
        enclosing box is still valid.
        """
        return (
            np.full(dimensions, -np.inf, dtype=float),
            np.full(dimensions, np.inf, dtype=float),
        )

    def max_feasible_step(
        self, x: NDArray[np.float64], direction: NDArray[np.float64]
    ) -> float:
        """
        Largest ``alpha >= 0`` such that ``x + alpha * direction`` is feasible.

        The ``stpmax`` a gradient line search expects.  Taken from
        :meth:`feasible_step_interval` when the handler provides one, else
        from the ratio test against :meth:`bounding_box`.  An unconstrained
        handler gets the full :data:`MAX_FEASIBLE_STEP`.
        """
        interval = self.feasible_step_interval(x, direction)
        if interval is None:
            lower, upper = self.bounding_box(len(x))
            interval = _box_step_interval(x, direction, lower, upper)
        return float(min(max(interval[1], 0.0), MAX_FEASIBLE_STEP))

    def project_direction(
        self, x: NDArray[np.float64], direction: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Drop the components of *direction* that immediately leave the feasible
        region from *x*, leaving the step along the still-feasible subspace.

        The default zeroes components pushing outward at an active box bound.
        Returns a new array, identical to the input when nothing is active.
        """
        lower, upper = self.bounding_box(len(x))
        projected = np.array(direction, dtype=np.float64, copy=True)
        blocked = ((x <= lower) & (projected < 0)) | ((x >= upper) & (projected > 0))
        projected[blocked] = 0.0
        return projected

    def projected_gradient(
        self, x: NDArray[np.float64], gradient: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        KKT first-order optimality measure at *x* — the gradient with the
        components that push into an active constraint removed.

        The default is the box projection
        (:func:`declivity.utils.optimality.projected_gradient`), which reduces
        to the plain gradient when the box is unbounded.
        """
        lower, upper = self.bounding_box(len(x))
        return _box_projected_gradient(x, gradient, lower, upper)

    def projected_gradient_inf_norm(
        self, x: NDArray[np.float64], gradient: NDArray[np.float64]
    ) -> float:
        """Infinity norm of :meth:`projected_gradient` (0.0 when empty)."""
        projected = self.projected_gradient(x, gradient)
        if len(projected) == 0:
            return 0.0
        return float(np.max(np.abs(projected)))

    def feasible_step_interval(
        self, x: NDArray[np.float64], direction: NDArray[np.float64]
    ) -> tuple[float, float] | None:
        """
        Return ``(alpha_min, alpha_max)`` such that ``x + alpha * direction``
        is feasible for every ``alpha`` in that closed interval, or ``None``.

        Returning ``None`` (the default) means the feasible set is not an
        interval along this ray, or cannot be computed cheaply.  Callers then
        search unconstrained and route every point through :meth:`repair`.

        Implement this for polytopes (boxes, linear constraints).  Leave the
        default for strictly convex regions, where every straight ray from a
        boundary point leaves immediately and the interval collapses to
        ``(0.0, 0.0)``.

        A degenerate ``(0.0, 0.0)`` means no non-zero step is feasible.
        """
        del x, direction
        return None


@final
class BoxConstraintHandler(ConstraintHandler):
    """
    Constraint handler for axis-aligned box constraints.

    Implements repair via either clamping or bounce-back (controlled by
    *strategy*).  Penalty is a no-op — box constraints are handled
    entirely through repair.
    """

    def __init__(
        self,
        strategy: BoxStrategy,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> None:
        """
        Parameters
        ----------
        strategy:
            ``BoxStrategy.CLAMP`` or ``BoxStrategy.BOUNCE_BACK``.
        lower_bounds:
            Per-dimension lower bound array.
        upper_bounds:
            Per-dimension upper bound array.
        """
        self.strategy: BoxStrategy = strategy
        self.lower_bounds: NDArray[np.float64] = lower_bounds
        self.upper_bounds: NDArray[np.float64] = upper_bounds

    @override
    def is_feasible(self, x: NDArray[np.float64]) -> bool:
        return bool(np.all(x >= self.lower_bounds)) and bool(
            np.all(x <= self.upper_bounds)
        )

    @override
    def feasibility_distance(self, x: NDArray[np.float64]) -> float:
        """Sum of squared bound violations (0.0 if feasible)."""
        lower_violation = np.maximum(0.0, self.lower_bounds - x)
        upper_violation = np.maximum(0.0, x - self.upper_bounds)
        return float(np.sum(lower_violation**2) + np.sum(upper_violation**2))

    @override
    def repair(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Repair *x* according to the selected box strategy."""
        if self.strategy is BoxStrategy.CLAMP:
            return self._repair_clamp(x)
        else:
            return self._repair_bounce_back(x)

    @override
    def repair_batch(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vectorised box repair for ``CLAMP``; ``BOUNCE_BACK`` is recursive
        and keeps the per-row fallback."""
        if self.strategy is BoxStrategy.CLAMP:
            sanitized = self._remove_inf_nan(population)
            return np.clip(sanitized, self.lower_bounds, self.upper_bounds)
        return super().repair_batch(population)

    @override
    def penalty(self, x: NDArray[np.float64], f_x: float) -> float:
        """No-op — box constraints are handled by repair, not penalty."""
        return f_x

    @override
    def bounding_box(
        self, dimensions: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The box itself."""
        if len(self.lower_bounds) != dimensions:
            raise ValueError(
                f"BoxConstraintHandler was built for "
                f"{len(self.lower_bounds)} dimensions but asked for "
                f"{dimensions}."
            )
        return self.lower_bounds, self.upper_bounds

    @override
    def feasible_step_interval(
        self, x: NDArray[np.float64], direction: NDArray[np.float64]
    ) -> tuple[float, float]:
        """Exact ray span for a box."""
        return _box_step_interval(x, direction, self.lower_bounds, self.upper_bounds)

    def _repair_clamp(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sanitize non-finite values, then clip into the box.

        Sanitizing comes first because ``np.clip`` propagates NaN.
        """
        x_repaired = self._remove_inf_nan(x)
        return np.clip(x_repaired, self.lower_bounds, self.upper_bounds)

    def _repair_bounce_back(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reflect out-of-bounds coordinates back into the box."""
        if self.is_feasible(x):
            return x

        x_repaired = x.copy()

        # Zero-width dimensions have no interior to bounce within and would
        # make the modulo below divide by zero.
        degenerate = self.upper_bounds == self.lower_bounds
        if np.any(degenerate):
            x_repaired[degenerate] = self.lower_bounds[degenerate]

        # Fix lower bound violations
        lower_violations = (x < self.lower_bounds) & ~degenerate
        if np.any(lower_violations):
            indices = np.where(lower_violations)[0]
            for i in indices:
                x_repaired[i] = self.lower_bounds[i] + (self.lower_bounds[i] - x[i]) % (
                    self.upper_bounds[i] - self.lower_bounds[i]
                )

        # Fix upper bound violations
        upper_violations = (x > self.upper_bounds) & ~degenerate
        if np.any(upper_violations):
            indices = np.where(upper_violations)[0]
            for i in indices:
                x_repaired[i] = self.upper_bounds[i] - (x[i] - self.upper_bounds[i]) % (
                    self.upper_bounds[i] - self.lower_bounds[i]
                )

        x_repaired = self._remove_inf_nan(x_repaired)

        if not self.is_feasible(x_repaired):
            return self._repair_bounce_back(x_repaired)

        return x_repaired

    def _remove_inf_nan(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Replace non-finite values with the extreme finite floats.

        NaN and ``+inf`` map to ``np.finfo(float).max``; ``-inf`` maps to
        ``np.finfo(float).min``, so a subsequent clip lands each value on the
        correct bound.

        Distinct from :func:`declivity.utils.helpers.delete_inf_nan`, which
        sends ``-inf`` to ``+DBL_MAX`` to match DES.R.
        """
        result = x.copy()
        result[np.isnan(result)] = np.finfo(float).max
        result[np.isposinf(result)] = np.finfo(float).max
        result[np.isneginf(result)] = np.finfo(float).min
        return result


class ConstraintHandlerType(Enum):
    """
    Discoverability enum listing all built-in constraint handlers.

    Call ``.build(lower_bounds, upper_bounds)`` to obtain a ready-to-use
    ``ConstraintHandler`` instance without importing concrete classes.
    """

    BOX_CLAMP = "box_clamp"
    BOX_BOUNCE_BACK = "box_bounce_back"

    def build(
        self,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> ConstraintHandler:
        """
        Construct the matching :class:`BoxConstraintHandler`.

        Parameters
        ----------
        lower_bounds:
            Per-dimension lower bound array.
        upper_bounds:
            Per-dimension upper bound array.

        Returns
        -------
        ConstraintHandler
            Concrete handler for this enum member.
        """
        if self is ConstraintHandlerType.BOX_CLAMP:
            return BoxConstraintHandler(BoxStrategy.CLAMP, lower_bounds, upper_bounds)
        elif self is ConstraintHandlerType.BOX_BOUNCE_BACK:
            return BoxConstraintHandler(
                BoxStrategy.BOUNCE_BACK, lower_bounds, upper_bounds
            )
        raise NotImplementedError(f"No build() implementation for {self!r}")
