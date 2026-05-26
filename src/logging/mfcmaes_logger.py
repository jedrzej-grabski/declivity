from typing import Any
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from src.algorithms.choices import AlgorithmChoice
from src.logging.base_logger import BaseLogger, PopulationLogData
from src.algorithms.mfcmaes.mfcmaes_config import MFCMAESConfig


@dataclass
class MFCMAESLogData(PopulationLogData):
    """MF-CMA-ES-specific log data."""

    sigma: list[float] = field(default_factory=list)
    """Step size values"""

    p_succ: list[float] = field(default_factory=list)
    """Success probability (PPMF)"""

    midpoint_fitness: list[float] = field(default_factory=list)
    """Fitness at distribution mean"""

    constraint_violations: list[int] = field(default_factory=list)
    """Number of constraint violations per generation"""

    pc: list[NDArray[np.float64]] = field(default_factory=list)
    """Evolution path for covariance"""

    pc_norm: list[float] = field(default_factory=list)
    """Norm of evolution path"""

    mean_vector: list[NDArray[np.float64]] = field(default_factory=list)
    """Distribution mean vector"""

    mean_vector_norm: list[float] = field(default_factory=list)
    """Norm of mean vector"""

    def clear(self) -> None:
        super().clear()
        self.sigma.clear()
        self.p_succ.clear()
        self.midpoint_fitness.clear()
        self.constraint_violations.clear()
        self.pc.clear()
        self.pc_norm.clear()
        self.mean_vector.clear()
        self.mean_vector_norm.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        return super().to_dict() | {
            "sigma": self.sigma,
            "p_succ": self.p_succ,
            "midpoint_fitness": self.midpoint_fitness,
            "constraint_violations": self.constraint_violations,
            "pc": self.pc,
            "pc_norm": self.pc_norm,
            "mean_vector": self.mean_vector,
            "mean_vector_norm": self.mean_vector_norm,
        }


class MFCMAESLogger(BaseLogger[MFCMAESLogData]):
    """Logger for MF-CMA-ES algorithm."""

    def __init__(self, config: MFCMAESConfig):
        super().__init__(config, AlgorithmChoice.MFCMAES)

    def _create_log_data(self) -> MFCMAESLogData:
        return MFCMAESLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        best_fitness: float,
        worst_fitness: float,
        mean_fitness: float,
        sigma: float,
        p_succ: float,
        midpoint_fitness: float,
        constraint_violations: int,
        fitness: NDArray[np.float64] | None = None,
        population: NDArray[np.float64] | None = None,
        best_solution: NDArray[np.float64] | None = None,
        pc: NDArray[np.float64] | None = None,
        mean_vector: NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        self.logs.iteration.append(iteration)
        self.logs.evaluations.append(evaluations)
        self.logs.best_fitness.append(best_fitness)
        self.logs.worst_fitness.append(worst_fitness)
        self.logs.mean_fitness.append(mean_fitness)
        self.logs.sigma.append(sigma)
        self.logs.p_succ.append(p_succ)
        self.logs.midpoint_fitness.append(midpoint_fitness)
        self.logs.constraint_violations.append(constraint_violations)

        if fitness is not None:
            self.logs.std_fitness.append(float(np.std(fitness)))
        else:
            self.logs.std_fitness.append(0.0)

        if population is not None and self.config.diag_pop:
            self.logs.population.append(population.copy())

        if best_solution is not None:
            self.logs.best_solution.append(best_solution.copy())

        # Evolution path
        if pc is not None:
            self.logs.pc.append(pc.copy())
            self.logs.pc_norm.append(float(np.linalg.norm(pc)))

        # Mean vector
        if mean_vector is not None:
            self.logs.mean_vector.append(mean_vector.copy())
            self.logs.mean_vector_norm.append(float(np.linalg.norm(mean_vector)))
