"""CEC benchmark problems, backed by cecxx's compiled evaluator.

Re-shares cecxx's own ``CECEdition`` enum instead of wrapping it in a second
one, so callers get real type hints for the supported editions straight from
cecxx and stay in sync with whatever it adds, without another wrapper layer.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from cecpy.benchmark import (
    CECEdition,
    CECEvaluator,
    benchmark_problem_num,
    benchmark_valid_dimensions,
    problem_optimum_value,
)
from numpy.typing import NDArray

from declivity.utils.benchmark_functions import BenchmarkFunction

CEC_LOWER_BOUND = -100.0
CEC_UPPER_BOUND = 100.0


@cache
def _evaluator(edition: CECEdition, dimensions: int) -> CECEvaluator:
    """One CECEvaluator per (edition, dimensions), shared by every CECProblem.

    Keyed on the dimension as well as the edition so only the shift and
    rotation tables actually needed get read off disk. Loading every valid
    dimension for CEC2017 costs ~0.26s against ~0.004s for a single one.
    """
    return CECEvaluator(edition, [dimensions])


class CECProblem(BenchmarkFunction):
    """A single (edition, function, dimensions) CEC benchmark problem.

    Not every CEC edition supports the same set of dimensionalities -- e.g.
    CEC2013 supports {2, 5, 10, 20, ..., 100} while CEC2014/CEC2017 only
    support {10, 30, 50, 100}.

    Gradients are not available.
    """

    def __init__(self, edition: CECEdition, function_number: int, dimensions: int):
        valid_dimensions = benchmark_valid_dimensions(edition)
        if dimensions not in valid_dimensions:
            raise ValueError(
                f"{edition.name} does not support dimensions={dimensions}; "
                f"supported dimensions are {valid_dimensions}."
            )
        n_problems = benchmark_problem_num(edition)
        if not (1 <= function_number <= n_problems):
            raise ValueError(
                f"{edition.name} function_number must be in [1, {n_problems}], "
                f"got {function_number}."
            )
        super().__init__(dimensions)
        self.edition = edition
        self.function_number = function_number
        self._evaluator = _evaluator(edition, dimensions)

    def __getstate__(self) -> dict[str, object]:
        # CECEvaluator is a nanobind extension object and cannot be pickled,
        # so it is dropped here and rebuilt from the cache on unpickling.
        # Without this, a CECProblem cannot cross a process boundary and
        # Benchmark(num_workers>1) fails.
        state = self.__dict__.copy()
        del state["_evaluator"]
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._evaluator = _evaluator(self.edition, self.dimensions)

    def __call__(self, x: NDArray[np.float64]) -> float:
        # cecxx's evaluator takes a (dimensions, num_points) matrix and
        # returns one result per column. The shape is explicit here because
        # the binding reads input.shape(1) unconditionally, so handing it a
        # 1D array is an out-of-bounds read that only happens to work.
        arr = np.ascontiguousarray(x, dtype=np.float64).reshape(-1, 1)
        # cecpy's stub declares Sequence[float]; the binding actually reads
        # shape(0) and shape(1) off the array, so the 2D form is required.
        return float(self._evaluator(self.function_number, arr)[0])  # pyright: ignore[reportArgumentType]

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            CEC_LOWER_BOUND * np.ones(self.dimensions),
            CEC_UPPER_BOUND * np.ones(self.dimensions),
        )

    @property
    def name(self) -> str:
        return f"{self.edition.name}-F{self.function_number}"

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        """Known optimal value only - CEC-based problems don't come with x_opt."""
        optimal_value = problem_optimum_value(self.edition, self.function_number)
        return np.full(self.dimensions, np.nan), float(optimal_value)
