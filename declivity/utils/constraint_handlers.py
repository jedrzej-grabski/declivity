"""
Constraint handler abstractions and box-constraint implementations.

Replaces the BoundaryHandler/BoundaryHandlerType strategy pattern with a
proper ABC that is open to inequality constraints, user-supplied callables,
and parameterised penalty functions in future slices.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import final, override

import numpy as np
from numpy.typing import NDArray


class BoxStrategy(Enum):
    """Repair strategy for a box-constrained domain."""

    CLAMP = "clamp"
    BOUNCE_BACK = "bounce_back"


class ConstraintHandler(ABC):
    """
    Abstract base class for constraint-handling strategies.

    A ConstraintHandler is responsible for four orthogonal concerns:

    * **Feasibility test** — ``is_feasible`` / ``feasibility_distance``
    * **Repair** — project or bounce an infeasible point back into the feasible
      region.  The default implementation is a no-op (returns *x* unchanged).
    * **Penalty** — augment an objective value to discourage infeasibility.
      The default implementation is a no-op (returns *f_x* unchanged).
    * **Directional feasibility** — ``feasible_step_interval``, the range of
      step lengths along a search ray that stays feasible.  Line-search
      optimizers (Powell) use it to confine the search *a priori* instead of
      repairing afterwards.  The default returns ``None`` ("cannot describe my
      feasible set along a ray"), which tells such an optimizer to fall back to
      repairing the points it accepts.

    Subclasses should override only the hooks they need.
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

    def feasible_step_interval(
        self, x: NDArray[np.float64], direction: NDArray[np.float64]
    ) -> tuple[float, float] | None:
        """
        Return ``(alpha_min, alpha_max)`` such that ``x + alpha * direction``
        is feasible for every ``alpha`` in that closed interval, or ``None``.

        This is the *a priori* counterpart of :meth:`repair`: a line-search
        optimizer that knows the feasible span of a ray never has to produce
        an infeasible point in the first place.  It is the mechanism by which
        Powell enforces constraints — see
        :class:`~declivity.algorithms.powell.powell_optimizer.PowellOptimizer`.

        Returning ``None`` (the default) means "my feasible set is not an
        interval along this ray, or I cannot compute one cheaply".  Callers
        must then search unconstrained and route every point they evaluate or
        accept through :meth:`repair` instead.

        **Override this for polytopes, not for curved sets.**  A box or a set
        of linear constraints has a boundary made of flat pieces, so a point on
        the boundary still has feasible directions to travel; confining the
        search is then better than repairing, because a repaired point is a
        different point from the one the caller asked for.  A *strictly convex*
        region (ball, ellipsoid) is the opposite case: every straight ray from a
        boundary point leaves immediately, so the interval collapses to
        ``(0.0, 0.0)`` and a line-search optimizer stalls the moment it touches
        the boundary.  Such handlers should keep the ``None`` default and let
        the caller project instead — projection can slide along the curve.

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

    # ------------------------------------------------------------------
    # ConstraintHandler interface
    # ------------------------------------------------------------------

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
        """Vectorised box repair.

        For ``CLAMP``, a single ``np.clip`` over the whole matrix
        replaces the per-row loop.  ``BOUNCE_BACK`` is recursive and
        keeps the per-row fallback.
        """
        if self.strategy is BoxStrategy.CLAMP:
            sanitized = self._remove_inf_nan(population)
            return np.clip(sanitized, self.lower_bounds, self.upper_bounds)
        return super().repair_batch(population)

    @override
    def penalty(self, x: NDArray[np.float64], f_x: float) -> float:
        """No-op — box constraints are handled by repair, not penalty."""
        return f_x

    @override
    def feasible_step_interval(
        self, x: NDArray[np.float64], direction: NDArray[np.float64]
    ) -> tuple[float, float]:
        """Ratio test for the span of a ray that stays inside the box.

        For each coordinate the direction actually moves, the two bounds give
        the step lengths at which that coordinate reaches them; the feasible
        interval is the intersection over coordinates.  Coordinates with a
        zero direction component never leave the box, so they are excluded.

        This is the same computation SciPy performs in
        ``scipy.optimize._optimize._line_for_search``, which is what keeps
        Powell's bounded trajectory bit-identical to SciPy's.  Both bounds may
        be infinite, in which case the corresponding end of the interval is
        infinite too.
        """
        (nonzero,) = np.asarray(direction).nonzero()
        if nonzero.size == 0:
            # A zero direction never moves: every step keeps x exactly where
            # it is, so no step length is excluded.
            return (-np.inf, np.inf)

        lower = self.lower_bounds[nonzero]
        upper = self.upper_bounds[nonzero]
        x_nz = x[nonzero]
        d_nz = direction[nonzero]
        low = (lower - x_nz) / d_nz
        high = (upper - x_nz) / d_nz

        # Moving in +d hits the upper bound last, in -d the lower bound;
        # ``where`` selects per coordinate without branching.
        pos = d_nz > 0
        alpha_min = float(np.max(np.where(pos, low, 0) + np.where(pos, 0, high)))
        alpha_max = float(np.min(np.where(pos, high, 0) + np.where(pos, 0, low)))

        return (alpha_min, alpha_max) if alpha_max >= alpha_min else (0.0, 0.0)

    # ------------------------------------------------------------------
    # Internal repair implementations
    # (ported character-for-character from BoundaryHandler subclasses)
    # ------------------------------------------------------------------

    def _repair_clamp(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clamp repair — sanitize non-finite values, then clip into the box.

        Sanitizing must come first: ``np.clip`` propagates NaN, and a
        post-clip replacement would land far outside the bounds.
        """
        x_repaired = self._remove_inf_nan(x)
        return np.clip(x_repaired, self.lower_bounds, self.upper_bounds)

    def _repair_bounce_back(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Bounce-back repair — mirrors BounceBackBoundaryHandler.repair (lines 66–106)."""
        if self.is_feasible(x):
            return x

        x_repaired = x.copy()

        # Zero-width dimensions have no interior to bounce within — the
        # modulo below would divide by zero (NaN).  Pin them to the unique
        # feasible value and exclude them from the bounce loops.
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

        # Handle any NaN or Inf values
        x_repaired = self._remove_inf_nan(x_repaired)

        # Recursively repair if still infeasible
        if not self.is_feasible(x_repaired):
            return self._repair_bounce_back(x_repaired)

        return x_repaired

    def _remove_inf_nan(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Replace non-finite values with the extreme finite floats.

        NaN and ``+inf`` map to ``np.finfo(float).max``; ``-inf`` maps to
        ``np.finfo(float).min`` — direction-preserving, so a subsequent
        clip lands each value on the correct bound.
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
        # Exhaustive match — new members must extend this method.
        raise NotImplementedError(f"No build() implementation for {self!r}")
