from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.logging.base_logger import BaseLogger, PopulationLogData
from declivity.logging.logger_factory import register_logger


@dataclass
class CMAESLogData(PopulationLogData):
    """CMA-ES-specific log data container."""

    # Basic fitness statistics
    median_fitness: list[float] = field(default_factory=list)
    """Median fitness per generation"""

    # Step-size and adaptation
    sigma: list[float] = field(default_factory=list)
    """Step size values"""

    # Evolution paths
    pc: list[NDArray[np.float64]] = field(default_factory=list)
    """Evolution path for covariance matrix"""

    ps: list[NDArray[np.float64]] = field(default_factory=list)
    """Evolution path for step size"""

    pc_norm: list[float] = field(default_factory=list)
    """Norm of covariance evolution path"""

    ps_norm: list[float] = field(default_factory=list)
    """Norm of step-size evolution path"""

    # Mean vector properties
    mean_vector: list[NDArray[np.float64]] = field(default_factory=list)
    """Mean vector (center of distribution)"""

    mean_vector_norm: list[float] = field(default_factory=list)
    """Norm of mean vector"""

    # NOTE: CMA-ES populates the inherited ``mean_fitness`` field with f(m) —
    # the fitness evaluated at the distribution mean, not the population
    # average. The "midpoint_fitness" semantic panel key in standard_panels
    # routes here so plots can label it correctly.

    # Covariance matrix properties
    covariance_determinant: list[float] = field(default_factory=list)
    """Determinant of covariance matrix"""

    max_eigenvalue: list[float] = field(default_factory=list)
    """Maximum eigenvalue"""

    min_eigenvalue: list[float] = field(default_factory=list)
    """Minimum eigenvalue"""

    # Coordinate-wise statistics
    coordinate_std: list[NDArray[np.float64]] = field(default_factory=list)
    """Standard deviation in each coordinate"""

    # Full covariance matrix (stored only for the final generation to avoid
    # excessive memory usage; use covariance_matrix[-1] after optimization)
    covariance_matrix: list[NDArray[np.float64]] = field(default_factory=list)
    """Full covariance matrix C (logged when diag_eigen is enabled)"""

    def clear(self) -> None:
        super().clear()
        self.median_fitness.clear()
        self.sigma.clear()
        self.pc.clear()
        self.ps.clear()
        self.pc_norm.clear()
        self.ps_norm.clear()
        self.mean_vector.clear()
        self.mean_vector_norm.clear()
        self.covariance_determinant.clear()
        self.max_eigenvalue.clear()
        self.min_eigenvalue.clear()
        self.coordinate_std.clear()
        self.covariance_matrix.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        return super().to_dict() | {
            "median_fitness": self.median_fitness,
            "sigma": self.sigma,
            "pc": self.pc,
            "ps": self.ps,
            "pc_norm": self.pc_norm,
            "ps_norm": self.ps_norm,
            "mean_vector": self.mean_vector,
            "mean_vector_norm": self.mean_vector_norm,
            "covariance_determinant": self.covariance_determinant,
            "max_eigenvalue": self.max_eigenvalue,
            "min_eigenvalue": self.min_eigenvalue,
            "coordinate_std": self.coordinate_std,
            "covariance_matrix": self.covariance_matrix,
        }


@register_logger(AlgorithmChoice.CMAES)
class CMAESLogger(BaseLogger[CMAESLogData]):
    """Logger for CMA-ES algorithm with proper typing."""

    def __init__(self, config: CMAESConfig):
        super().__init__(config, AlgorithmChoice.CMAES)

    def _create_log_data(self) -> CMAESLogData:
        """Create CMA-ES-specific log data container."""
        return CMAESLogData()

    def log_iteration(
        self,
        iteration: int,
        evaluations: int,
        sigma: float = 0.0,
        fitness: NDArray[np.float64] | None = None,
        population: NDArray[np.float64] | None = None,
        best_fitness: float = float("inf"),
        worst_fitness: float = float("inf"),
        best_solution: NDArray[np.float64] | None = None,
        mean_fitness: float = 0.0,
        median_fitness: float = 0.0,
        pc: NDArray[np.float64] | None = None,
        ps: NDArray[np.float64] | None = None,
        mean_vector: NDArray[np.float64] | None = None,
        eigenvalues: NDArray[np.float64] | None = None,
        covariance_matrix: NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        """Log CMA-ES iteration data."""

        self.logs.iteration.append(iteration)
        self.logs.evaluations.append(evaluations)
        self.logs.best_fitness.append(best_fitness)
        self.logs.worst_fitness.append(worst_fitness)
        self.logs.mean_fitness.append(mean_fitness)
        self.logs.median_fitness.append(median_fitness)

        if fitness is not None:
            self.logs.std_fitness.append(float(np.std(fitness)))
        else:
            self.logs.std_fitness.append(0.0)

        if population is not None and self.config.diag_pop:
            self.logs.population.append(population.copy())

        if best_solution is not None:
            self.logs.best_solution.append(best_solution.copy())

        # Eigenvalues and condition number
        if eigenvalues is not None and self.config.diag_eigen:
            self.logs.eigenvalues.append(eigenvalues.copy())
            if len(eigenvalues) > 0:
                self.logs.condition_number.append(eigenvalues[-1] / eigenvalues[0])
                self.logs.max_eigenvalue.append(float(eigenvalues[-1]))
                self.logs.min_eigenvalue.append(float(eigenvalues[0]))

        # Step-size
        if self.config.diag_sigma:
            self.logs.sigma.append(sigma)

        # Evolution paths
        if pc is not None:
            self.logs.pc.append(pc.copy())
            self.logs.pc_norm.append(float(np.linalg.norm(pc)))

        if ps is not None:
            self.logs.ps.append(ps.copy())
            self.logs.ps_norm.append(float(np.linalg.norm(ps)))

        # Mean vector
        if mean_vector is not None:
            self.logs.mean_vector.append(mean_vector.copy())
            self.logs.mean_vector_norm.append(float(np.linalg.norm(mean_vector)))

        # Covariance matrix properties
        if covariance_matrix is not None and self.config.diag_eigen:
            det = np.linalg.det(covariance_matrix)
            self.logs.covariance_determinant.append(float(det))

            coord_std = np.sqrt(np.diag(covariance_matrix))
            self.logs.coordinate_std.append(coord_std)

            if self.config.diag_covariance_matrix:
                # Store every generation (expensive but allows tracking evolution)
                self.logs.covariance_matrix.append(covariance_matrix.copy())
            else:
                # Store only the latest generation to keep memory bounded
                if len(self.logs.covariance_matrix) > 0:
                    self.logs.covariance_matrix[-1] = covariance_matrix.copy()
                else:
                    self.logs.covariance_matrix.append(covariance_matrix.copy())
