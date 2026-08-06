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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from declivity.utils.benchmark_functions import BenchmarkFunction
from declivity.utils.initial_point_generator import (
    FixedInitialPointGenerator,
    InitialPointGenerator,
    UniformInitialPointGenerator,
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

    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
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
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        initial_point_generator: InitialPointGenerator
        | NDArray[np.float64]
        | None = None,
    ) -> Problem:
        """Build a Problem from a BenchmarkFunction.

        Picks up the function's bounds and ``gradient`` attribute (if any)
        unless explicit values are passed. ``initial_point_generator`` is
        forwarded verbatim (``None`` → uniform sampling within the bounds).
        """
        lb_array, ub_array = function.bounds
        # Only advertise an analytic gradient when the concrete function
        # overrides BenchmarkFunction.gradient — the base method is a stub
        # that raises NotImplementedError, so a bare getattr would hand
        # L-BFGS-B a gradient that blows up on first call instead of letting
        # it fall back to finite differences.
        overrides_gradient = (
            getattr(type(function), "gradient", None) is not BenchmarkFunction.gradient
        )
        return cls(
            name=name,
            function=function,
            dimensions=function.dimensions,
            lower_bound=float(lb_array[0]) if lower_bound is None else lower_bound,
            upper_bound=float(ub_array[0]) if upper_bound is None else upper_bound,
            gradient=function.gradient if overrides_gradient else None,
            initial_point_generator=initial_point_generator,
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
