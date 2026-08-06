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

    A ConstraintHandler is responsible for three orthogonal concerns:

    * **Feasibility test** — ``is_feasible`` / ``feasibility_distance``
    * **Repair** — project or bounce an infeasible point back into the feasible
      region.  The default implementation is a no-op (returns *x* unchanged).
    * **Penalty** — augment an objective value to discourage infeasibility.
      The default implementation is a no-op (returns *f_x* unchanged).

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

    def repair_batch(
        self, population: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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
    def repair_batch(
        self, population: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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
                x_repaired[i] = self.lower_bounds[i] + (
                    self.lower_bounds[i] - x[i]
                ) % (self.upper_bounds[i] - self.lower_bounds[i])

        # Fix upper bound violations
        upper_violations = (x > self.upper_bounds) & ~degenerate
        if np.any(upper_violations):
            indices = np.where(upper_violations)[0]
            for i in indices:
                x_repaired[i] = self.upper_bounds[i] - (
                    x[i] - self.upper_bounds[i]
                ) % (self.upper_bounds[i] - self.lower_bounds[i])

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
