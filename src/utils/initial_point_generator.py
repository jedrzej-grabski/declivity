"""Pluggable initial-point generation for optimization problems.

Two concrete implementations ship by default:
- UniformInitialPointGenerator: reproduces Problem.starting_point's previous
  rng.uniform(lower, upper, size=dim) behavior exactly (bit-identical RNG).
- FixedInitialPointGenerator: wraps a pre-supplied NDArray and returns a copy,
  ignoring RNG and bounds.

Pick via the InitialPointGeneratorType discoverability enum:
    ipg = InitialPointGeneratorType.UNIFORM.build()
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class InitialPointGenerator(ABC):
    """ABC for single-point initial-position generators.

    Subclasses must implement ``generate_point``.  The method receives
    a seeded RNG and the full bounds arrays so implementations can be
    both reproducible and bounds-aware without carrying state.
    """

    @abstractmethod
    def generate_point(
        self,
        rng: np.random.Generator,
        dimensions: int,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return a single starting point of shape ``(dimensions,)``.

        Args:
            rng: A seeded NumPy Generator (``np.random.default_rng(seed)``).
            dimensions: Number of decision variables.
            lower_bounds: Per-dimension lower bounds, shape ``(dimensions,)``.
            upper_bounds: Per-dimension upper bounds, shape ``(dimensions,)``.

        Returns:
            Starting point array of shape ``(dimensions,)``.
        """


class UniformInitialPointGenerator(InitialPointGenerator):
    """Samples uniformly at random within the box.

    Calls ``rng.uniform(lower_bounds, upper_bounds, size=dimensions)``,
    which is bit-identical to the previous Problem.starting_point()
    implementation even when ``lower_bounds`` / ``upper_bounds`` are
    passed as full NDArrays instead of scalars (NumPy broadcasts
    identically for scalar vs array arguments to rng.uniform).
    """

    def generate_point(
        self,
        rng: np.random.Generator,
        dimensions: int,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return rng.uniform(lower_bounds, upper_bounds, size=dimensions)


class FixedInitialPointGenerator(InitialPointGenerator):
    """Always returns the same pre-supplied point.

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
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.point.copy()


class InitialPointGeneratorType(Enum):
    """Discoverability enum for built-in InitialPointGenerator implementations.

    Usage::

        ipg = InitialPointGeneratorType.UNIFORM.build()
        x0  = ipg.generate_point(rng, dim, lb, ub)
    """

    UNIFORM = "uniform"
    FIXED = "fixed"

    def build(self, **kwargs: object) -> InitialPointGenerator:
        """Construct the corresponding InitialPointGenerator.

        For UNIFORM no arguments are needed.
        For FIXED pass ``point=<NDArray>``.

        Raises:
            ValueError: If FIXED is requested without a ``point`` argument.
        """
        if self is InitialPointGeneratorType.UNIFORM:
            return UniformInitialPointGenerator()
        if self is InitialPointGeneratorType.FIXED:
            if "point" not in kwargs:
                raise ValueError(
                    "InitialPointGeneratorType.FIXED.build() requires a 'point' keyword argument"
                )
            return FixedInitialPointGenerator(kwargs["point"])  # type: ignore[arg-type]
        raise ValueError(f"Unknown InitialPointGeneratorType: {self}")
