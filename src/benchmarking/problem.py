"""A test problem: function, dimensions, bounds, and a deterministic
starting-point generator keyed by seed.

Two algorithms run with the same seed start from the same x0 — that's the
property the framework relies on for fair side-by-side comparison.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from src.utils.benchmark_functions import BenchmarkFunction


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
        )

    def starting_point(self, seed: int) -> NDArray[np.float64]:
        """Deterministic uniform starting point inside the bounds.

        Same seed => same x0, regardless of which algorithm consumes it.
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(self.lower_bound, self.upper_bound, size=self.dimensions)
