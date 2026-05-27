"""
Population-level repair strategy abstractions.

A RepairStrategy operates on an entire population matrix (n_individuals,
n_dimensions) and produces a repaired matrix of the same shape.  It
delegates the per-individual constraint logic to an injected
ConstraintHandler, keeping repair policy separate from feasibility
semantics.

Relationship to ConstraintHandler
----------------------------------
* ``ConstraintHandler`` decides *what is feasible* and how to repair a
  **single point**.
* ``RepairStrategy`` decides *how to apply that repair across a
  population* (or whether to skip it entirely).

The split means that L-BFGS-B (single-point) never needs a
RepairStrategy, while evolutionary algorithms can swap policies without
touching their ConstraintHandler.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import final, override

import numpy as np
from numpy.typing import NDArray

from src.utils.constraint_handlers import BoxConstraintHandler, ConstraintHandler


class RepairStrategy(ABC):
    """
    Abstract base class for population-level repair strategies.

    A ``RepairStrategy`` receives a population matrix of shape
    ``(n_individuals, n_dimensions)`` — each **row** is one individual —
    and returns a matrix of the same shape with infeasible individuals
    repaired.

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
        Return a repaired copy of *population*.

        Parameters
        ----------
        population:
            Array of shape ``(n_individuals, n_dimensions)``.  Rows are
            individuals.
        constraint_handler:
            The constraint handler whose ``repair()`` method is used for
            per-individual repair (except for optimised vectorised paths).

        Returns
        -------
        NDArray[np.float64]
            Array of the same shape as *population* with infeasible
            individuals repaired.
        """
        ...


@final
class IdentityRepair(RepairStrategy):
    """
    No-op repair — returns the population unchanged.

    Intended for CMA-ES, where the reference port handles boundary
    correction internally and a second repair pass would break numerical
    equivalence.
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
    Per-individual repair via ``constraint_handler.repair()``.

    Applies ``constraint_handler.repair()`` to every row in *population*.
    For ``BoxConstraintHandler`` this goes through
    ``_repair_clamp`` → ``_remove_inf_nan``, exactly matching the
    boundary-handling behaviour in the current DES implementation.
    """

    @override
    def repair_population(
        self,
        population: NDArray[np.float64],
        constraint_handler: ConstraintHandler,
    ) -> NDArray[np.float64]:
        return np.array(
            [constraint_handler.repair(individual) for individual in population]
        )


@final
class ClampRepair(RepairStrategy):
    """
    Vectorised clamp repair.

    Fast path: if *constraint_handler* is a ``BoxConstraintHandler``,
    delegates to ``np.clip`` directly and skips the ``_remove_inf_nan``
    pass.  This matches the current MF-CMA-ES ``np.clip`` behaviour,
    preserving numerical equivalence.

    Fallback: for any other ``ConstraintHandler``, falls back to
    per-individual ``constraint_handler.repair()``.

    .. note::
       The deliberate absence of ``_remove_inf_nan`` in the fast path is
       intentional — see the critical equivalence note in T01 plan.
    """

    @override
    def repair_population(
        self,
        population: NDArray[np.float64],
        constraint_handler: ConstraintHandler,
    ) -> NDArray[np.float64]:
        if isinstance(constraint_handler, BoxConstraintHandler):
            return np.clip(
                population,
                constraint_handler.lower_bounds,
                constraint_handler.upper_bounds,
            )
        # Generic fallback — per-individual repair
        return np.array(
            [constraint_handler.repair(individual) for individual in population]
        )


class RepairStrategyType(Enum):
    """
    Discoverability enum listing all built-in repair strategies.

    Call ``.build()`` to obtain a ready-to-use ``RepairStrategy`` instance
    without importing concrete classes directly.

    Members
    -------
    IDENTITY
        No-op repair.  Use for CMA-ES (reference port handles boundaries
        internally).
    LAMARCKIAN
        Per-individual repair via ``ConstraintHandler.repair()``.  Use
        for DES (includes ``_remove_inf_nan`` for box constraints).
    CLAMP
        Vectorised ``np.clip`` for box constraints, per-individual fallback
        otherwise.  Use for MF-CMA-ES.
    """

    IDENTITY = "identity"
    LAMARCKIAN = "lamarckian"
    CLAMP = "clamp"

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
        elif self is RepairStrategyType.CLAMP:
            return ClampRepair()
        # Exhaustive match — new members must extend this method.
        raise NotImplementedError(f"No build() implementation for {self!r}")
