"""CEC benchmark problems, backed by cecxx.

Re-shares cecxx's own ``CECEdition`` enum (rather than wrapping it) so the
supported editions come with real type hints and stay in sync with whatever
cecxx supports.

    from declivity.cec import CECEdition, CECProblem

    problem = CECProblem(CECEdition.CEC2017, function_number=1, dimensions=10)

Usually built through ``Problem.from_cec`` instead, which wraps the result
in a ``declivity.benchmarking.Problem``.
"""

from cecpy.benchmark import (
    CECEdition,
    benchmark_problem_num,
    benchmark_valid_dimensions,
)

from declivity.cec.problem import CECProblem

__all__ = [
    "CECEdition",
    "CECProblem",
    "benchmark_problem_num",
    "benchmark_valid_dimensions",
]
