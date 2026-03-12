"""
A collection of benchmark functions for testing optimization algorithms.
"""

import numpy as np
from numpy.typing import NDArray

from opfunu.cec_based import cec2017


class BenchmarkFunction:
    """Base class for benchmark functions."""

    def __init__(self, dimensions: int):
        """
        Initialize a benchmark function.

        Args:
            dimensions: Number of dimensions for the function
        """
        self.dimensions = dimensions

    def __call__(self, x: NDArray[np.float64]) -> float:
        """
        Evaluate the function at point x.

        Args:
            x: Input vector of length self.dimensions

        Returns:
            Function value at x
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the bounds of the function.

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        """
        Get the global minimum of the function.

        Returns:
            Tuple of (optimal_solution, optimal_value)
        """
        raise NotImplementedError("Subclasses must implement this method")


class Sphere(BenchmarkFunction):
    """
    Sphere function.
    f(x) = sum(x_i^2)
    Global minimum: f(0, 0, ..., 0) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        return np.sum(x**2)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -100.0 * np.ones(self.dimensions), 100.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class Ellipsoid(BenchmarkFunction):
    """
    Ellipsoid function.
    f(x) = sum(10^(6*(i-1)/(n-1)) * x_i^2)
    Global minimum: f(0, 0, ..., 0) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        n = len(x)
        if n == 1:
            return float(x[0] ** 2)
        scales = 10.0 ** (6.0 * np.arange(n) / (n - 1))
        return float(np.sum(scales * x**2))

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -100.0 * np.ones(self.dimensions), 100.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class Rosenbrock(BenchmarkFunction):
    """
    Rosenbrock function (Valley function).
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum: f(1, 1, ..., 1) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -5.0 * np.ones(self.dimensions), 10.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.ones(self.dimensions), 0.0


class Rastrigin(BenchmarkFunction):
    """
    Rastrigin function.
    f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum: f(0, 0, ..., 0) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        return 10 * self.dimensions + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -5.12 * np.ones(self.dimensions), 5.12 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class Ackley(BenchmarkFunction):
    """
    Ackley function.
    f(x) = -20*exp(-0.2*sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e
    Global minimum: f(0, 0, ..., 0) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
        return term1 + term2 + 20.0 + np.e

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -32.768 * np.ones(self.dimensions), 32.768 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class Schwefel(BenchmarkFunction):
    """
    Schwefel function.
    f(x) = 418.9829*d - sum(x_i * sin(sqrt(abs(x_i))))
    Global minimum: f(420.9687, 420.9687, ..., 420.9687) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        return 418.9829 * self.dimensions - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -500.0 * np.ones(self.dimensions), 500.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return 420.9687 * np.ones(self.dimensions), 0.0


class CEC17Function(BenchmarkFunction):

    def __init__(self, dimensions: int, function_id: int):
        """
        Initialize a CEC benchmark function.

        Args:
            dimensions: Number of dimensions for the function
            function_id: ID of the CEC function to use
        """
        super().__init__(dimensions)

        if function_id < 1 or function_id > 30:
            raise ValueError("Function ID must be between 1 and 29.")

        self.function_id = function_id

        fname = f"F{function_id}2017"
        self.func = getattr(cec2017, fname)(dimensions)

    def __call__(self, x: NDArray[np.float64]) -> float:
        """
        Evaluate the CEC function at point x.

        Args:
            x: Input vector of length self.dimensions

        Returns:
            Function value at x
        """
        return self.func.evaluate(x)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the bounds of the CEC function.

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        return self.func.lower, self.func.upper

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        """
        Get the global minimum of the CEC function.

        Returns:
            Tuple of (optimal_solution, optimal_value)
        """
        return self.func.optimal_solution, self.func.optimal_value
