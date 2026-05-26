from typing import Any
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from src.algorithms.choices import AlgorithmChoice
from src.logging.base_logger import BaseLogger, BaseLogData


@dataclass
class LBFGSBLogData(BaseLogData):
    """L-BFGS-B-specific log data."""

    function_value: list[float] = field(default_factory=list)
    """Current function value at each iteration."""

    gradient_norm: list[float] = field(default_factory=list)
    """L2 norm of the gradient."""

    projected_gradient_norm: list[float] = field(default_factory=list)
    """Infinity norm of the projected gradient (convergence measure)."""

    step_length: list[float] = field(default_factory=list)
    """Line search step length accepted at each iteration."""

    theta: list[float] = field(default_factory=list)
    """L-BFGS scaling factor (y'y / y's)."""

    num_free_vars: list[int] = field(default_factory=list)
    """Number of free (non-active) variables at each iteration."""

    num_corrections: list[int] = field(default_factory=list)
    """Number of L-BFGS correction pairs stored."""

    line_search_iters: list[int] = field(default_factory=list)
    """Number of function evaluations in the line search."""

    def clear(self) -> None:
        super().clear()
        self.function_value.clear()
        self.gradient_norm.clear()
        self.projected_gradient_norm.clear()
        self.step_length.clear()
        self.theta.clear()
        self.num_free_vars.clear()
        self.num_corrections.clear()
        self.line_search_iters.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        return super().to_dict() | {
            "function_value": self.function_value,
            "gradient_norm": self.gradient_norm,
            "projected_gradient_norm": self.projected_gradient_norm,
            "step_length": self.step_length,
            "theta": self.theta,
            "num_free_vars": self.num_free_vars,
            "num_corrections": self.num_corrections,
            "line_search_iters": self.line_search_iters,
        }


class LBFGSBLogger(BaseLogger[LBFGSBLogData]):
    """Logger for L-BFGS-B algorithm."""

    def __init__(self, config):
        super().__init__(config, AlgorithmChoice.LBFGSB)

    def _create_log_data(self) -> LBFGSBLogData:
        return LBFGSBLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        best_fitness: float,
        function_value: float,
        gradient_norm: float = 0.0,
        projected_gradient_norm: float = 0.0,
        step_length: float = 0.0,
        theta: float = 1.0,
        num_free: int = 0,
        num_corrections: int = 0,
        line_search_iters: int = 0,
        best_solution: NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        self.logs.iteration.append(iteration)
        self.logs.evaluations.append(evaluations)
        self.logs.best_fitness.append(best_fitness)
        self.logs.function_value.append(function_value)

        if self.config.diag_gradient_norm:
            self.logs.gradient_norm.append(gradient_norm)
            self.logs.projected_gradient_norm.append(projected_gradient_norm)

        if self.config.diag_step_length:
            self.logs.step_length.append(step_length)

        if self.config.diag_theta:
            self.logs.theta.append(theta)

        if self.config.diag_num_free:
            self.logs.num_free_vars.append(num_free)
            self.logs.num_corrections.append(num_corrections)

        if self.config.diag_line_search_iters:
            self.logs.line_search_iters.append(line_search_iters)

        if best_solution is not None:
            self.logs.best_solution.append(best_solution.copy())
