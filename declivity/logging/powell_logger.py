from typing import Any
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from declivity.algorithms.choices import AlgorithmChoice
from declivity.logging.base_logger import BaseLogger, BaseLogData
from declivity.logging.logger_factory import register_logger


@dataclass
class PowellLogData(BaseLogData):
    """Powell-specific log data.

    Powell is single-point, so this extends :class:`BaseLogData`
    directly (like L-BFGS-B).  One record per outer iteration — i.e.
    per full sweep over the direction set — logged at the same boundary
    where SciPy fires its per-iteration callback, so traces compare
    one-to-one against ``scipy.optimize.minimize(method='Powell')``.
    """

    function_value: list[float] = field(default_factory=list)
    """Function value at the end of each sweep."""

    delta: list[float] = field(default_factory=list)
    """Largest single-direction decrease achieved during the sweep."""

    big_direction_index: list[int] = field(default_factory=list)
    """Index of the direction that achieved ``delta``."""

    direction_replaced: list[int] = field(default_factory=list)
    """1 if the direction set was updated at the end of the *previous*
    iteration (so this sweep ran with a fresh direction), else 0."""

    step_norm: list[float] = field(default_factory=list)
    """Displacement norm ||x_k - x_{k-1}|| between consecutive logged
    iterates (includes any end-of-iteration extrapolation line search,
    matching SciPy's per-callback displacement)."""

    line_search_evals: list[int] = field(default_factory=list)
    """Function evaluations spent in line searches during the sweep."""

    direc_condition_number: list[float] = field(default_factory=list)
    """2-norm condition number of the direction-set matrix."""

    direc_determinant: list[float] = field(default_factory=list)
    """|det| of the direction-set matrix (1.0 for the identity start;
    decays as replacements make the set less orthogonal)."""

    direction_set: list[NDArray[np.float64]] = field(default_factory=list)
    """Full (n, n) direction matrices (only with ``diag_direc_matrix``)."""

    current_point: list[NDArray[np.float64]] = field(default_factory=list)
    """The iterate x_k at the end of each sweep — the trajectory itself,
    as opposed to ``best_solution`` (best-ever).  Matches the ``x`` SciPy
    hands to its per-iteration callback."""

    def clear(self) -> None:
        super().clear()
        self.function_value.clear()
        self.delta.clear()
        self.big_direction_index.clear()
        self.direction_replaced.clear()
        self.step_norm.clear()
        self.line_search_evals.clear()
        self.direc_condition_number.clear()
        self.direc_determinant.clear()
        self.direction_set.clear()
        self.current_point.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        return super().to_dict() | {
            "function_value": self.function_value,
            "delta": self.delta,
            "big_direction_index": self.big_direction_index,
            "direction_replaced": self.direction_replaced,
            "step_norm": self.step_norm,
            "line_search_evals": self.line_search_evals,
            "direc_condition_number": self.direc_condition_number,
            "direc_determinant": self.direc_determinant,
            "direction_set": self.direction_set,
            "current_point": self.current_point,
        }


@register_logger(AlgorithmChoice.POWELL)
class PowellLogger(BaseLogger[PowellLogData]):
    """Logger for Powell's method."""

    def __init__(self, config):
        super().__init__(config, AlgorithmChoice.POWELL)

    def _create_log_data(self) -> PowellLogData:
        return PowellLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        best_fitness: float,
        function_value: float,
        delta: float = 0.0,
        big_direction_index: int = 0,
        direction_replaced: bool = False,
        step_norm: float = 0.0,
        line_search_evals: int = 0,
        direc_condition_number: float = 0.0,
        direc_determinant: float = 0.0,
        direction_set: NDArray[np.float64] | None = None,
        best_solution: NDArray[np.float64] | None = None,
        current_point: NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        self.logs.iteration.append(iteration)
        self.logs.evaluations.append(evaluations)
        self.logs.best_fitness.append(best_fitness)
        self.logs.function_value.append(function_value)

        if self.config.diag_delta:
            self.logs.delta.append(delta)
            self.logs.big_direction_index.append(big_direction_index)
            self.logs.direction_replaced.append(int(direction_replaced))

        if self.config.diag_step_length:
            self.logs.step_norm.append(step_norm)

        if self.config.diag_line_search_iters:
            self.logs.line_search_evals.append(line_search_evals)

        if self.config.diag_direc:
            self.logs.direc_condition_number.append(direc_condition_number)
            self.logs.direc_determinant.append(direc_determinant)

        if direction_set is not None and self.config.diag_direc_matrix:
            self.logs.direction_set.append(direction_set.copy())

        if best_solution is not None:
            self.logs.best_solution.append(best_solution.copy())

        if current_point is not None:
            self.logs.current_point.append(current_point.copy())
