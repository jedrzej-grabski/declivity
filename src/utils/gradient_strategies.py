"""
Gradient-computation strategies for derivative-based optimisers.

Each concrete class encapsulates how the gradient ``∇f(x)`` is
approximated from black-box function evaluations, making the choice
pluggable without touching the optimiser body.  Optimisers receive a
strategy at construction time and call its
:meth:`~GradientStrategy.compute` method whenever they need a gradient
— exactly the same pattern as :class:`~src.utils.repair_strategies.RepairStrategy`
and :class:`~src.utils.population_initializers.PopulationInitializer`.

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
from typing import Callable

import numpy as np
from numpy.typing import NDArray


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
            and save one evaluation per gradient; symmetric schemes
            ignore it.

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
    ) -> NDArray[np.float64]:
        if f_at_x is None:
            f_at_x = f(x)
        num_vars = len(x)
        gradient = np.zeros(num_vars)
        for i in range(num_vars):
            x_forward = x.copy()
            x_forward[i] += eps
            gradient[i] = (f(x_forward) - f_at_x) / eps
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
    ) -> NDArray[np.float64]:
        del f_at_x  # symmetric scheme does not reuse the centre value
        num_vars = len(x)
        gradient = np.zeros(num_vars)
        for i in range(num_vars):
            x_forward = x.copy()
            x_backward = x.copy()
            x_forward[i] += eps
            x_backward[i] -= eps
            gradient[i] = (f(x_forward) - f(x_backward)) / (2.0 * eps)
        return gradient


