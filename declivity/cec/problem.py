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
)
from numpy.typing import NDArray

from declivity.utils.benchmark_functions import BenchmarkFunction

CEC_LOWER_BOUND = -100.0
CEC_UPPER_BOUND = 100.0


@cache
def _evaluator(edition: CECEdition) -> CECEvaluator:
    """One CECEvaluator per edition, shared by every CECProblem. Avoids redundant repeated disk IO by caching."""
    return CECEvaluator(edition, benchmark_valid_dimensions(edition))


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
        self._evaluator = _evaluator(edition)

    def __call__(self, x: NDArray[np.float64]) -> float:
        # cecxx's evaluator always returns one result per input column; a
        # bare 1D array of length `dimensions` counts as a single column.
        arr = np.ascontiguousarray(x, dtype=np.float64)
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
