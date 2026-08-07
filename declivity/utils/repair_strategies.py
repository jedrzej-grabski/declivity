"""
Population-level repair strategy abstractions.

A RepairStrategy is the *policy* layer that sits above
:class:`~declivity.utils.constraint_handlers.ConstraintHandler`: it decides
whether to repair a population at all, leaving the per-point (and
per-batch) projection mechanics to the handler.

Relationship to ConstraintHandler
----------------------------------
* ``ConstraintHandler`` defines *mechanism* — what is feasible, and
  how to project a single point (``repair``) or a whole population
  (``repair_batch``) onto the feasible region.
* ``RepairStrategy`` defines *policy* — does the optimiser apply
  ``repair_batch`` to its λ candidates this generation, or carry them
  through unrepaired?

Together the split keeps L-BFGS-B (single-point, no policy needed)
free of strategy machinery while letting evolutionary algorithms swap
policy without touching the handler.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from declivity.utils.constraint_handlers import ConstraintHandler


class RepairStrategy(ABC):
    """
    Abstract base class for population-level repair strategies.

    A ``RepairStrategy`` receives a population matrix of shape
    ``(n_individuals, n_dimensions)`` — each **row** is one individual —
    and returns a matrix of the same shape.  Concrete strategies are
    pure delegators: they decide whether to call
    :meth:`ConstraintHandler.repair_batch` on the population, not how
    that batch repair is computed.

    Implementations must be stateless: the same instance must produce
    identical output given the same inputs.
    """

    @abstractmethod
    def repair_population(
        self,
        population: NDArray[np.float64],
        constraint_handler: ConstraintHandler,
    ) -> NDArray[np.float64]:
        """
        Return a (possibly repaired) copy of *population*.

        Parameters
        ----------
        population:
            Array of shape ``(n_individuals, n_dimensions)``.  Rows are
            individuals.
        constraint_handler:
            The constraint handler that provides
            :meth:`~ConstraintHandler.repair_batch` for the actual
            per-row projection.

        Returns
        -------
        NDArray[np.float64]
            Array of the same shape as *population*.
        """
        ...


@final
class IdentityRepair(RepairStrategy):
    """
    No-op repair — returns the population unchanged.

    Use when the algorithm needs to carry possibly-infeasible
    individuals through (e.g., to compute its own penalty term) rather
    than projecting them onto the feasible region.
    """

    @override
    def repair_population(
        self,
        population: NDArray[np.float64],
        constraint_handler: ConstraintHandler,
    ) -> NDArray[np.float64]:
        return population


@final
class LamarckianRepair(RepairStrategy):
    """
    Apply repair to every row of the population.

    Delegates straight through to
    :meth:`ConstraintHandler.repair_batch`.  For
    :class:`~declivity.utils.constraint_handlers.BoxConstraintHandler` with
    ``BoxStrategy.CLAMP`` that is a single vectorised ``np.clip`` plus
    a NaN/Inf strip; for other handlers it falls back to the per-row
    default in the ABC.
    """

    @override
    def repair_population(
        self,
        population: NDArray[np.float64],
        constraint_handler: ConstraintHandler,
    ) -> NDArray[np.float64]:
        return constraint_handler.repair_batch(population)


class RepairStrategyType(Enum):
    """
    Discoverability enum listing all built-in repair strategies.

    Call ``.build()`` to obtain a ready-to-use ``RepairStrategy``
    instance without importing concrete classes directly.

    Members
    -------
    IDENTITY
        No-op repair.  Carries the population through unmodified — the
        algorithm is responsible for any penalty term.
    LAMARCKIAN
        Apply ``ConstraintHandler.repair_batch`` to every individual.
        The default for every evolutionary algorithm in the framework.
    """

    IDENTITY = "identity"
    LAMARCKIAN = "lamarckian"

    def build(self) -> RepairStrategy:
        """
        Construct and return a fresh ``RepairStrategy`` instance.

        Returns
        -------
        RepairStrategy
            A concrete repair strategy for this enum member.
        """
        if self is RepairStrategyType.IDENTITY:
            return IdentityRepair()
        elif self is RepairStrategyType.LAMARCKIAN:
            return LamarckianRepair()
        raise NotImplementedError(f"No build() implementation for {self!r}")
