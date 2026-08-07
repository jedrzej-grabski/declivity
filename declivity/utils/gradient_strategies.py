"""
Gradient-computation strategies for derivative-based optimisers.

Each concrete class encapsulates how the gradient ``∇f(x)`` is
approximated from black-box function evaluations, making the choice
pluggable without touching the optimiser body.  Optimisers receive a
strategy at construction time and call its
:meth:`~GradientStrategy.compute` method whenever they need a gradient
— exactly the same pattern as :class:`~declivity.utils.repair_strategies.RepairStrategy`
and :class:`~declivity.utils.population_initializers.PopulationInitializer`.

Hierarchy
---------
- :class:`GradientStrategy` — abstract base (single abstract method)
- :class:`ForwardFD` — one-sided 2-point difference, ``N+1`` evals, ``O(ε)`` error
- :class:`CentralFD` — symmetric 2-point difference, ``2N`` evals, ``O(ε²)`` error

The strategy accepts a *callable* ``f`` rather than a reference to the
owning optimiser so that evaluation-count bookkeeping stays a concern
of the caller: pass in the optimiser's ``evaluate`` method, and every
function evaluation made by the strategy will increment the budget
counter naturally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, override

import numpy as np
from numpy.typing import NDArray

from declivity.utils.constraint_handlers import ConstraintHandler


class GradientStrategy(ABC):
    """Abstract base class for gradient-computation strategies.

    Concrete subclasses implement :meth:`compute`, returning the
    estimated gradient at ``x``.  Implementations must use ``f`` (a
    caller-supplied evaluator) for every function value they need so
    that evaluation budgets are accounted for outside the strategy.
    """

    @abstractmethod
    def compute(
        self,
        f: Callable[[NDArray[np.float64]], float],
        x: NDArray[np.float64],
        eps: float,
        f_at_x: float | None = None,
        constraint_handler: ConstraintHandler | None = None,
    ) -> NDArray[np.float64]:
        """Approximate ``∇f(x)``.

        Parameters
        ----------
        f:
            Evaluator that returns the (penalised) objective value at a
            single point.  Strategies must route every required
            evaluation through this callable so that evaluation-budget
            accounting stays in the caller's hands.
        x:
            Point at which the gradient is sought, shape ``(dim,)``.
        eps:
            Finite-difference step.  Strategies may interpret this
            differently (e.g. central FD uses ``eps`` symmetrically;
            forward FD uses ``eps`` one-sided).
        f_at_x:
            Optional cached value of ``f(x)``.  Forward FD can reuse it
            and save one evaluation per gradient; central FD needs it
            only for coordinates that fall back to a one-sided
            difference at a bound.
        constraint_handler:
            Optional
            :class:`~declivity.utils.constraint_handlers.ConstraintHandler`
            defining the feasible region.  When supplied, a perturbed
            point the handler rejects switches that coordinate to a
            one-sided difference on the feasible side (standard practice
            for constrained solvers whose iterates sit exactly on a
            bound).  ``None`` — or probes the handler accepts —
            reproduces the unconstrained scheme exactly.  The handler is
            the only source of feasibility here; the strategy never
            inspects bound arrays itself.

        Returns
        -------
        NDArray[np.float64]
            Estimated gradient, shape ``(dim,)``.
        """
        ...


class ForwardFD(GradientStrategy):
    """One-sided 2-point finite difference.

    Computes ``gᵢ = (f(x + ε·eᵢ) − f(x)) / ε`` for each coordinate
    ``i``.  Total cost: ``N + 1`` evaluations (or ``N`` if ``f(x)`` is
    supplied via ``f_at_x``).  Truncation error is ``O(ε)`` from the
    Taylor expansion ``f(x+ε) = f(x) + ε·f'(x) + O(ε²)``.

    Cheaper than central FD but less accurate near a stationary point,
    where central FD's symmetric cancellation of even-order terms wins.
    This is scipy's default for ``scipy.optimize.minimize`` when no
    analytical gradient is supplied.
    """

    def compute(
        self,
        f: Callable[[NDArray[np.float64]], float],
        x: NDArray[np.float64],
        eps: float,
        f_at_x: float | None = None,
        constraint_handler: ConstraintHandler | None = None,
    ) -> NDArray[np.float64]:
        if f_at_x is None:
            f_at_x = f(x)
        num_vars = len(x)
        gradient = np.zeros(num_vars)
        for i in range(num_vars):
            step = eps
            if constraint_handler is not None:
                forward = x.copy()
                forward[i] += eps
                if not constraint_handler.is_feasible(forward):
                    backward = x.copy()
                    backward[i] -= eps
                    if constraint_handler.is_feasible(backward):
                        # Forward probe leaves the feasible region:
                        # difference backward instead (one-sided on the
                        # feasible side).
                        step = -eps
            x_probe = x.copy()
            x_probe[i] += step
            gradient[i] = (f(x_probe) - f_at_x) / step
        return gradient


class CentralFD(GradientStrategy):
    """Symmetric 2-point finite difference (framework default).

    Computes ``gᵢ = (f(x + ε·eᵢ) − f(x − ε·eᵢ)) / (2ε)`` for each
    coordinate ``i``.  Total cost: ``2N`` evaluations.  Truncation
    error is ``O(ε²)`` because odd-order terms of the Taylor expansion
    cancel pairwise.

    Twice as expensive per gradient as :class:`ForwardFD` but
    substantially more accurate, particularly in the late phase of
    convergence when ``‖∇f(x)‖`` is small and forward-FD truncation
    becomes the dominant error.
    """

    def compute(
        self,
        f: Callable[[NDArray[np.float64]], float],
        x: NDArray[np.float64],
        eps: float,
        f_at_x: float | None = None,
        constraint_handler: ConstraintHandler | None = None,
    ) -> NDArray[np.float64]:
        num_vars = len(x)
        gradient = np.zeros(num_vars)
        for i in range(num_vars):
            x_forward = x.copy()
            x_backward = x.copy()
            x_forward[i] += eps
            x_backward[i] -= eps
            if constraint_handler is None:
                forward_ok = backward_ok = True
            else:
                forward_ok = constraint_handler.is_feasible(x_forward)
                backward_ok = constraint_handler.is_feasible(x_backward)
            if forward_ok == backward_ok:
                # Both probes feasible: symmetric difference.  (Also the
                # degenerate fallback when neither side fits.)
                gradient[i] = (f(x_forward) - f(x_backward)) / (2.0 * eps)
            elif forward_ok:
                # Backward probe leaves the feasible region: one-sided
                # forward, anchored on the (lazily evaluated) centre value.
                if f_at_x is None:
                    f_at_x = f(x)
                gradient[i] = (f(x_forward) - f_at_x) / eps
            else:
                if f_at_x is None:
                    f_at_x = f(x)
                gradient[i] = (f_at_x - f(x_backward)) / eps
        return gradient


class _RayRestrictedHandler(ConstraintHandler):
    """The feasible region seen by the 1-D restriction ``phi(t) = f(x + t*d)``.

    A :class:`GradientStrategy` asks its handler whether a *perturbed point*
    is feasible.  When the strategy is differentiating along a ray, the point
    it perturbs is the scalar ``t``, not the vector it maps to — so the real
    handler cannot be passed straight through.  This adapter translates each
    feasibility question back into the full space and forwards it, which keeps
    the feasible region the run's *actual* handler's decision rather than a
    box re-derived at the call site.
    """

    def __init__(
        self,
        handler: ConstraintHandler,
        x: NDArray[np.float64],
        direction: NDArray[np.float64],
    ) -> None:
        self._handler = handler
        self._x = x
        self._direction = direction

    def _point(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._x + float(t[0]) * self._direction

    @override
    def is_feasible(self, x: NDArray[np.float64]) -> bool:
        return self._handler.is_feasible(self._point(x))

    @override
    def feasibility_distance(self, x: NDArray[np.float64]) -> float:
        return self._handler.feasibility_distance(self._point(x))


def directional_derivative(
    strategy: GradientStrategy,
    f: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    direction: NDArray[np.float64],
    eps: float,
    f_at_x: float | None = None,
    constraint_handler: ConstraintHandler | None = None,
) -> float:
    """Approximate ``phi'(0)`` for ``phi(t) = f(x + t * direction)``.

    The ``phi'(alpha)`` a :class:`~declivity.utils.line_search.gradient.GradientLineSearch`
    needs, obtained by handing the *same injected*
    :class:`GradientStrategy` the 1-D restriction of the objective along the
    ray.  Routing it through the strategy is what keeps a single injected
    component in charge of *all* finite differencing in a run: differencing
    ``∇f`` with central differences while the line search silently used its own
    hardcoded scheme would make the ``gradient_strategy=`` seam only half live.

    ``constraint_handler`` is wrapped in a :class:`_RayRestrictedHandler`, so a
    probe that would leave the feasible region switches that difference to the
    feasible side exactly as it does for a coordinate gradient — relevant
    because a line search is allowed to land *on* the boundary
    (``max_feasible_step``), which puts the outward probe outside it.

    With ``CentralFD`` (the framework default) and ``f_at_x`` supplied this
    costs the same two evaluations as a hand-rolled central difference, in the
    same order.
    """
    ray_handler = (
        None
        if constraint_handler is None
        else _RayRestrictedHandler(constraint_handler, x, direction)
    )
    gradient = strategy.compute(
        f=lambda t: f(x + float(t[0]) * direction),
        x=np.zeros(1, dtype=float),
        eps=eps,
        f_at_x=f_at_x,
        constraint_handler=ray_handler,
    )
    return float(gradient[0])


class GradientStrategyType(Enum):
    """
    Discoverability enum listing all built-in gradient strategies.

    Call ``.build()`` to obtain a ready-to-use :class:`GradientStrategy`
    instance without importing concrete classes.
    """

    FORWARD_FD = "forward_fd"
    CENTRAL_FD = "central_fd"

    def build(self) -> GradientStrategy:
        """
        Construct the matching :class:`GradientStrategy`.

        Returns
        -------
        GradientStrategy
            Concrete strategy for this enum member.
        """
        if self is GradientStrategyType.FORWARD_FD:
            return ForwardFD()
        elif self is GradientStrategyType.CENTRAL_FD:
            return CentralFD()
        # Exhaustive match — new members must extend this method.
        raise NotImplementedError(f"No build() implementation for {self!r}")
