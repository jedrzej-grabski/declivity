"""
Constraint handler abstractions and box-constraint implementations.

A :class:`ConstraintHandler` is the **single authority on the feasible
region**.  Nothing else in the framework decides what "in bounds" means:
optimizers, population initializers, and gradient strategies all ask the
handler, so swapping the handler swaps the geometry everywhere at once.

The interface is grouped by what an optimizer needs to know:

===========================  ==================================================
Question                     Method
===========================  ==================================================
Is this point allowed?       :meth:`~ConstraintHandler.is_feasible`,
                             :meth:`~ConstraintHandler.feasibility_distance`
Put it back inside.          :meth:`~ConstraintHandler.repair`,
                             :meth:`~ConstraintHandler.repair_batch`
Punish it instead.           :meth:`~ConstraintHandler.penalty`
What box encloses me?        :meth:`~ConstraintHandler.bounding_box`
How far along this ray?      :meth:`~ConstraintHandler.feasible_step_interval`,
                             :meth:`~ConstraintHandler.max_feasible_step`
Which way can I still move?  :meth:`~ConstraintHandler.project_direction`,
                             :meth:`~ConstraintHandler.projected_gradient`
===========================  ==================================================

Only the first group is abstract.  Everything in the *geometry* half has a
default derived from :meth:`~ConstraintHandler.bounding_box`, so a subclass
that declares its enclosing box gets correct ray spans, direction
projections, and KKT measures for free — and a subclass that declares
nothing behaves as unconstrained, which is the right answer for a handler
that only penalises.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from declivity.utils.optimality import projected_gradient as _box_projected_gradient

MAX_FEASIBLE_STEP = 1e10
"""Cap returned by :meth:`ConstraintHandler.max_feasible_step` for a ray that
is unbounded in the search direction.  Line searches want a finite ``stpmax``;
this is large enough never to bind on a real problem."""


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
    component never leave the box, so they are excluded.

    This is the same computation SciPy performs in
    ``scipy.optimize._optimize._line_for_search``.  Either bound may be
    infinite, in which case the matching end of the interval is infinite too.
    A returned ``(0.0, 0.0)`` means no non-zero step is feasible.
    """
    (nonzero,) = np.asarray(direction).nonzero()
    if nonzero.size == 0:
        # A zero direction never moves: no step length is excluded.
        return (-np.inf, np.inf)

    lower = lower_bounds[nonzero]
    upper = upper_bounds[nonzero]
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


class BoxStrategy(Enum):
    """Repair strategy for a box-constrained domain."""

    CLAMP = "clamp"
    BOUNCE_BACK = "bounce_back"


class ConstraintHandler(ABC):
    """
    Abstract base class for constraint-handling strategies.

    A ConstraintHandler is responsible for five orthogonal concerns:

    * **Feasibility test** — ``is_feasible`` / ``feasibility_distance``
    * **Repair** — project or bounce an infeasible point back into the feasible
      region.  The default implementation is a no-op (returns *x* unchanged).
    * **Penalty** — augment an objective value to discourage infeasibility.
      The default implementation is a no-op (returns *f_x* unchanged).
    * **Enclosing box** — ``bounding_box``, the tightest axis-aligned box
      containing the feasible region.  Coordinate-wise algorithms (L-BFGS-B's
      Cauchy point, finite-difference probes, population sampling) need a box
      and cannot work with an arbitrary region; this is how they get one.
      Defaults to unbounded.
    * **Directional feasibility** — ``feasible_step_interval`` /
      ``max_feasible_step`` / ``project_direction`` / ``projected_gradient``:
      how far a search ray may travel, and which directions are still
      available at an active constraint.  All four default to the answers
      implied by ``bounding_box``.

    Subclasses should override only the hooks they need — declaring
    ``bounding_box`` alone gives correct defaults for the whole geometry half.
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

    # ------------------------------------------------------------------
    # Geometry — everything below has a default derived from bounding_box.
    # ------------------------------------------------------------------

    def bounding_box(
        self, dimensions: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Tightest axis-aligned box ``(lower, upper)`` containing the feasible
        region, as two arrays of length *dimensions*.

        This is how the framework asks "what are the bounds?" — there is no
        other source.  Algorithms that are *structurally* box-based (L-BFGS-B's
        generalized Cauchy point walks per-coordinate breakpoints; population
        initializers scale their spread by the search range; finite-difference
        probes must not step outside) read the box from here.

        The default is unbounded (``-inf``/``+inf``), which is correct for a
        handler that constrains nothing or expresses its constraints purely as
        a :meth:`penalty`.  A handler with a bounded feasible set should
        override this even when the set is not a box — a *conservative*
        enclosing box still gives the coordinate-wise algorithms something
        valid to work with, and the sharper hooks below can refine it.
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

        The scalar ``stpmax`` a gradient line search expects.  Taken from
        :meth:`feasible_step_interval` when the handler provides one; a handler
        that returns ``None`` there falls back to the ratio test against
        :meth:`bounding_box`.  That fallback matters: it is only a *cap* (the
        accepted point is still repaired), so using the conservative enclosing
        box keeps the line search from wandering arbitrarily far outside the
        feasible region — while an unconstrained handler, whose box is
        infinite, still gets the full :data:`MAX_FEASIBLE_STEP`.
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
        region from *x*.

        At an active constraint a search direction can have components that
        cannot move at all.  Leaving them in makes
        :meth:`max_feasible_step` zero and strands the optimizer even though
        the remaining coordinates are free; removing them yields the step
        along the still-feasible subspace.

        The default zeroes components pushing outward at an active *box* bound.
        Returns a new array; identical to the input when nothing is active
        (in particular, always so for an unbounded handler).
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

        This is the quantity a constrained optimizer must drive to zero: at a
        constrained optimum the raw gradient need not vanish, only its feasible
        part.  The default is the box projection
        (:func:`declivity.utils.optimality.projected_gradient`), which reduces
        *identically* to the plain gradient when the box is unbounded — so an
        optimizer testing this reproduces an unconstrained ``‖grad‖`` test
        exactly.
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
    def bounding_box(
        self, dimensions: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The box itself — this handler *is* its bounding box."""
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
        """Exact ray span for a box — the shared :func:`_box_step_interval`.

        Because the box *is* the feasible region, the ratio test is exact here
        rather than a conservative enclosure, and it reproduces SciPy's
        ``_line_for_search``, which is what keeps Powell's bounded trajectory
        bit-identical to SciPy's.
        """
        return _box_step_interval(x, direction, self.lower_bounds, self.upper_bounds)

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
