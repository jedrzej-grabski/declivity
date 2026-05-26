from typing import Any
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from src.algorithms.choices import AlgorithmChoice
from src.logging.base_logger import BaseLogger, PopulationLogData
from src.algorithms.des.config import DESConfig
from src.logging.logger_factory import register_logger


@dataclass
class DESLogData(PopulationLogData):
    """DES-specific log data container."""

    ft: list[float] = field(default_factory=list)
    evolution_path: list[NDArray[np.float64]] = field(default_factory=list)
    step_size: list[float] = field(default_factory=list)

    def clear(self) -> None:
        super().clear()
        self.ft.clear()
        self.evolution_path.clear()
        self.step_size.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        return super().to_dict() | {
            "ft": self.ft,
            "evolution_path": self.evolution_path,
            "step_size": self.step_size,
        }


@register_logger(AlgorithmChoice.DES)
class DESLogger(BaseLogger[DESLogData]):
    """Logger for DES algorithm with proper typing."""

    def __init__(self, config: DESConfig):
        super().__init__(config, AlgorithmChoice.DES)

    def _create_log_data(self) -> DESLogData:
        """Create DES-specific log data container."""
        return DESLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        ft: float = 0.0,
        fitness: NDArray[np.float64] | None = None,
        population: NDArray[np.float64] | None = None,
        best_fitness: float = float("inf"),
        worst_fitness: float = float("inf"),
        best_solution: NDArray[np.float64] | None = None,
        mean_fitness: float = 0.0,
        evolution_path: NDArray[np.float64] | None = None,
        eigenvalues: NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        """Log DES iteration data."""

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
                self.logs.condition_number.append(eigenvalues[0] / eigenvalues[-1])

        if self.config.diag_ft:
            self.logs.ft.append(ft)

        if evolution_path is not None:
            self.logs.evolution_path.append(evolution_path.copy())

        self.logs.step_size.append(ft)
