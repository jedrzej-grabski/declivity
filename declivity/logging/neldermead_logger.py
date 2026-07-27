from typing import Any
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from declivity.algorithms.choices import AlgorithmChoice
from declivity.logging.base_logger import BaseLogger, PopulationLogData
from declivity.logging.logger_factory import register_logger


@dataclass
class NelderMeadLogData(PopulationLogData):
    """Nelder-Mead-specific log data.

    The simplex is the population, so this extends
    :class:`PopulationLogData`: ``population`` holds the full simplex
    (with ``diag_pop``), ``worst/mean/std_fitness`` describe the vertex
    fitness distribution, and ``eigenvalues`` / ``condition_number``
    describe the vertex-covariance geometry (with ``diag_eigen``).
    """

    simplex_diameter: list[float] = field(default_factory=list)
    """Max elementwise extent ``max |sim[1:] - sim[0]|`` — the quantity
    tested against ``xatol``. The Nelder-Mead analog of a step size."""

    fitness_spread: list[float] = field(default_factory=list)
    """Max fitness gap to the best vertex ``max |f_0 - f_i|`` — the
    quantity tested against ``fatol``."""

    simplex_volume: list[float] = field(default_factory=list)
    """Simplex volume |det(edges)| / n! (only with ``diag_volume``)."""

    operation: list[int] = field(default_factory=list)
    """Simplex operation performed each iteration
    (:class:`~declivity.algorithms.neldermead.neldermead_optimizer.SimplexOperation`
    codes; only with ``diag_operations``)."""

    def clear(self) -> None:
        super().clear()
        self.simplex_diameter.clear()
        self.fitness_spread.clear()
        self.simplex_volume.clear()
        self.operation.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        return super().to_dict() | {
            "simplex_diameter": self.simplex_diameter,
            "fitness_spread": self.fitness_spread,
            "simplex_volume": self.simplex_volume,
            "operation": self.operation,
        }


@register_logger(AlgorithmChoice.NELDERMEAD)
class NelderMeadLogger(BaseLogger[NelderMeadLogData]):
    """Logger for the Nelder-Mead simplex method."""

    def __init__(self, config):
        super().__init__(config, AlgorithmChoice.NELDERMEAD)

    def _create_log_data(self) -> NelderMeadLogData:
        return NelderMeadLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        best_fitness: float = float("inf"),
        worst_fitness: float = float("inf"),
        mean_fitness: float = 0.0,
        fitness: NDArray[np.float64] | None = None,
        population: NDArray[np.float64] | None = None,
        best_solution: NDArray[np.float64] | None = None,
        simplex_diameter: float = 0.0,
        fitness_spread: float = 0.0,
        operation: int = 0,
        simplex_volume: float = 0.0,
        eigenvalues: NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        self.logs.iteration.append(iteration)
        self.logs.evaluations.append(evaluations)
        self.logs.best_fitness.append(best_fitness)
        self.logs.worst_fitness.append(worst_fitness)
        self.logs.mean_fitness.append(mean_fitness)

        if fitness is not None:
            self.logs.std_fitness.append(float(np.std(fitness)))
        else:
            self.logs.std_fitness.append(0.0)

        if population is not None and self.config.diag_pop:
            self.logs.population.append(population.copy())

        if best_solution is not None:
            self.logs.best_solution.append(best_solution.copy())

        if eigenvalues is not None and self.config.diag_eigen:
            self.logs.eigenvalues.append(eigenvalues.copy())
            if len(eigenvalues) > 0:
                self.logs.condition_number.append(
                    float(eigenvalues[0] / eigenvalues[-1])
                )

        self.logs.simplex_diameter.append(simplex_diameter)
        self.logs.fitness_spread.append(fitness_spread)

        if self.config.diag_operations:
            self.logs.operation.append(operation)

        if self.config.diag_volume:
            self.logs.simplex_volume.append(simplex_volume)
