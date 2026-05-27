"""A test problem: function, dimensions, bounds, and a deterministic
starting-point generator keyed by seed.

Two algorithms run with the same seed start from the same x0 — that's the
property the framework relies on for fair side-by-side comparison.

The ``initial_point_generator`` field is pluggable:
- Pass ``None`` (default) → ``UniformInitialPointGenerator`` (reproduces
  the previous ``rng.uniform`` behavior exactly, bit-identical).
- Pass an ``NDArray`` → wrapped in ``FixedInitialPointGenerator`` so the
  same fixed x0 is returned for every seed.
- Pass any ``InitialPointGenerator`` subclass directly for custom strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from src.utils.benchmark_functions import BenchmarkFunction
from src.utils.initial_point_generator import (
    InitialPointGenerator,
    UniformInitialPointGenerator,
    FixedInitialPointGenerator,
)


@dataclass
class Problem:
    """Test problem specification."""

    name: str
    """Short name used in plots and tables."""

    function: Callable[[NDArray[np.float64]], float]
    """Objective; usually a BenchmarkFunction subclass."""

    dimensions: int

    lower_bound: float
    upper_bound: float

    gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None
    """Analytical gradient. If None, L-BFGS-B uses finite differences."""

    initial_point_generator: InitialPointGenerator | NDArray[np.float64] | None = None
    """Strategy for generating the starting point.

    - ``None`` (default): uniform sampling within ``[lower_bound, upper_bound]``.
    - ``NDArray``: always start from this fixed point regardless of seed.
    - ``InitialPointGenerator`` subclass: delegate to its ``generate_point``.
    """

    def __post_init__(self) -> None:
        # Normalise the initial_point_generator field so it is always an
        # InitialPointGenerator instance after construction.
        if self.initial_point_generator is None:
            self.initial_point_generator = UniformInitialPointGenerator()
        elif isinstance(self.initial_point_generator, np.ndarray):
            self.initial_point_generator = FixedInitialPointGenerator(
                self.initial_point_generator
            )
        # Otherwise it is already an InitialPointGenerator instance — leave it.

    @classmethod
    def from_benchmark(
        cls,
        name: str,
        function: BenchmarkFunction,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> "Problem":
        """Build a Problem from a BenchmarkFunction.

        Picks up the function's bounds and ``gradient`` attribute (if any)
        unless explicit values are passed.
        """
        lb_array, ub_array = function.bounds
        return cls(
            name=name,
            function=function,
            dimensions=function.dimensions,
            lower_bound=float(lb_array[0]) if lower_bound is None else lower_bound,
            upper_bound=float(ub_array[0]) if upper_bound is None else upper_bound,
            gradient=getattr(function, "gradient", None),
            # initial_point_generator defaults to None → UniformInitialPointGenerator
        )

    def starting_point(self, seed: int) -> NDArray[np.float64]:
        """Deterministic starting point inside the bounds.

        Same seed => same x0, regardless of which algorithm consumes it.
        Delegates to ``self.initial_point_generator.generate_point``.
        """
        rng = np.random.default_rng(seed)
        lower_bounds = np.full(self.dimensions, self.lower_bound)
        upper_bounds = np.full(self.dimensions, self.upper_bound)
        assert isinstance(self.initial_point_generator, InitialPointGenerator)
        return self.initial_point_generator.generate_point(
            rng, self.dimensions, lower_bounds, upper_bounds
        )
