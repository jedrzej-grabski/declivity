"""Pluggable initial-point generation for optimization problems.

Two concrete implementations ship by default:
- UniformInitialPointGenerator: ``rng.uniform(lower, upper, size=dim)`` over
  the handler's bounding box.
- FixedInitialPointGenerator: wraps a pre-supplied NDArray and returns a copy,
  ignoring RNG and the feasible region.

Pick via the InitialPointGeneratorType discoverability enum:
    ipg = InitialPointGeneratorType.UNIFORM.build()

A generator learns about the feasible region from a
:class:`~declivity.utils.constraint_handlers.ConstraintHandler` rather than
from bound arrays.
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from declivity.utils.constraint_handlers import ConstraintHandler


class InitialPointGenerator(ABC):
    """ABC for single-point initial-position generators.

    Subclasses must implement ``generate_point``.  The method receives a
    seeded RNG and the run's constraint handler, so implementations can be
    both reproducible and feasibility-aware without carrying state.
    """

    @abstractmethod
    def generate_point(
        self,
        rng: np.random.Generator,
        dimensions: int,
        constraint_handler: ConstraintHandler,
    ) -> NDArray[np.float64]:
        """Return a single starting point of shape ``(dimensions,)``.

        Args:
            rng: A seeded NumPy Generator (``np.random.default_rng(seed)``).
            dimensions: Number of decision variables.
            constraint_handler: The run's feasible region.  Use
                ``bounding_box(dimensions)`` for a range to sample from, and
                ``is_feasible`` / ``repair`` for a region that is not a box.

        Returns:
            Starting point array of shape ``(dimensions,)``.
        """


class UniformInitialPointGenerator(InitialPointGenerator):
    """Samples uniformly at random within the handler's bounding box.

    Calls ``rng.uniform(lower_bounds, upper_bounds, size=dimensions)`` on the
    box the handler declares.  For a non-box feasible region the sample is
    drawn from the enclosing box and then repaired.
    """

    def generate_point(
        self,
        rng: np.random.Generator,
        dimensions: int,
        constraint_handler: ConstraintHandler,
    ) -> NDArray[np.float64]:
        lower_bounds, upper_bounds = constraint_handler.bounding_box(dimensions)
        point = rng.uniform(lower_bounds, upper_bounds, size=dimensions)
        if constraint_handler.is_feasible(point):
            return point
        return constraint_handler.repair(point)


class UniformBoxInitialPointGenerator(InitialPointGenerator):
    """Samples uniformly from a fixed box, ignoring the feasible region.

    For when the initial-mean region must differ from the feasible box, e.g.
    reproducing a reference that draws the starting mean from
    ``U[-100, 100]^d`` over an asymmetric feasible box such as
    ``[-180, 20]^d``.  The returned point may be infeasible.

    Args:
        lower: Lower corner of the sampling box (broadcast over dimensions).
        upper: Upper corner of the sampling box.
    """

    def __init__(self, lower: float, upper: float) -> None:
        self.lower = float(lower)
        self.upper = float(upper)

    def generate_point(
        self,
        rng: np.random.Generator,
        dimensions: int,
        constraint_handler: ConstraintHandler,
    ) -> NDArray[np.float64]:
        del constraint_handler  # ignored; see the class docstring
        return rng.uniform(self.lower, self.upper, size=dimensions)


class FixedInitialPointGenerator(InitialPointGenerator):
    """Always returns the same pre-supplied point.

    The point is returned verbatim, without a feasibility check.  Each
    optimizer repairs its own starting point.

    Args:
        point: The fixed starting point to return.  A copy is returned on
               every call so callers cannot mutate the stored array.
    """

    def __init__(self, point: NDArray[np.float64]) -> None:
        self.point = np.asarray(point, dtype=np.float64)

    def generate_point(
        self,
        rng: np.random.Generator,
        dimensions: int,
        constraint_handler: ConstraintHandler,
    ) -> NDArray[np.float64]:
        del rng, dimensions, constraint_handler
        return self.point.copy()


class InitialPointGeneratorType(Enum):
    """Discoverability enum for built-in InitialPointGenerator implementations.

    Usage::

        ipg = InitialPointGeneratorType.UNIFORM.build()
        x0  = ipg.generate_point(rng, dim, constraint_handler)
    """

    UNIFORM = "uniform"
    UNIFORM_BOX = "uniform_box"
    FIXED = "fixed"

    def build(self, **kwargs: object) -> InitialPointGenerator:
        """Construct the corresponding InitialPointGenerator.

        For UNIFORM no arguments are needed.
        For UNIFORM_BOX pass ``lower=<float>`` and ``upper=<float>``.
        For FIXED pass ``point=<NDArray>``.

        Raises:
            ValueError: If required keyword arguments are missing.
        """
        if self is InitialPointGeneratorType.UNIFORM:
            return UniformInitialPointGenerator()
        if self is InitialPointGeneratorType.UNIFORM_BOX:
            if "lower" not in kwargs or "upper" not in kwargs:
                raise ValueError(
                    "InitialPointGeneratorType.UNIFORM_BOX.build() requires "
                    "'lower' and 'upper' keyword arguments"
                )
            return UniformBoxInitialPointGenerator(
                float(kwargs["lower"]),  # type: ignore[arg-type]
                float(kwargs["upper"]),  # type: ignore[arg-type]
            )
        if self is InitialPointGeneratorType.FIXED:
            if "point" not in kwargs:
                raise ValueError(
                    "InitialPointGeneratorType.FIXED.build() requires a 'point' keyword argument"
                )
            return FixedInitialPointGenerator(kwargs["point"])  # type: ignore[arg-type]
        raise ValueError(f"Unknown InitialPointGeneratorType: {self}")
