from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.logging.base_logger import BaseLogData, BaseLogger
from declivity.logging.logger_factory import register_logger


@dataclass
class BFGSLogData(BaseLogData):
    """BFGS-specific log data.

    BFGS is single-point, so this extends :class:`BaseLogData` directly
    (like L-BFGS-B and Powell).  One record per accepted iteration —
    logged at the same boundary where SciPy fires its per-iteration
    callback, so traces compare one-to-one against
    ``scipy.optimize.minimize(method='BFGS')``.
    """

    function_value: list[float] = field(default_factory=list)
    """Current function value at each iteration."""

    gradient_norm: list[float] = field(default_factory=list)
    """Norm of the gradient used for the convergence test."""

    step_length: list[float] = field(default_factory=list)
    """Line-search step length accepted at each iteration."""

    curvature: list[float] = field(default_factory=list)
    """BFGS curvature ``yk . sk`` (positive keeps ``Hk`` positive definite)."""

    hessian_condition: list[float] = field(default_factory=list)
    """Condition number of the inverse-Hessian ``Hk``."""

    def clear(self) -> None:
        super().clear()
        self.function_value.clear()
        self.gradient_norm.clear()
        self.step_length.clear()
        self.curvature.clear()
        self.hessian_condition.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        return super().to_dict() | {
            "function_value": self.function_value,
            "gradient_norm": self.gradient_norm,
            "step_length": self.step_length,
            "curvature": self.curvature,
            "hessian_condition": self.hessian_condition,
        }


@register_logger(AlgorithmChoice.BFGS)
class BFGSLogger(BaseLogger[BFGSLogData]):
    """Logger for the BFGS algorithm."""

    def __init__(self, config):
        super().__init__(config, AlgorithmChoice.BFGS)

    def _create_log_data(self) -> BFGSLogData:
        return BFGSLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        best_fitness: float,
        function_value: float,
        gradient_norm: float = 0.0,
        step_length: float = 0.0,
        curvature: float = 0.0,
        hessian_condition: float = 0.0,
        best_solution: NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        self.logs.iteration.append(iteration)
        self.logs.evaluations.append(evaluations)
        self.logs.best_fitness.append(best_fitness)
        self.logs.function_value.append(function_value)

        if self.config.diag_gradient_norm:
            self.logs.gradient_norm.append(gradient_norm)

        if self.config.diag_step_length:
            self.logs.step_length.append(step_length)

        if self.config.diag_curvature:
            self.logs.curvature.append(curvature)

        if self.config.diag_hessian_condition:
            self.logs.hessian_condition.append(hessian_condition)

        if best_solution is not None:
            self.logs.best_solution.append(best_solution.copy())
