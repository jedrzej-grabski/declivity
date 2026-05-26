"""LogData hierarchy + the abstract logger that owns one.

Three layers:

- :class:`BaseLogData`       — universal fields every algorithm logs
                               (iteration, evaluations, best_fitness, best_solution).
- :class:`PopulationLogData` — adds worst / mean / std fitness, the
                               population itself, eigenvalues, condition
                               number. Extended by DES / CMA-ES / MF-CMA-ES.
- algorithm-specific data    — extends one of the two above and adds
                               algorithm-specific fields (sigma, Ft, theta,
                               gradient_norm, ...).

L-BFGS-B is a single-point method, so :class:`LBFGSBLogData` extends
:class:`BaseLogData` directly and never inherits the population fields.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, Generic

import numpy as np
from numpy.typing import NDArray

from src.algorithms.choices import AlgorithmChoice

LogDataType = TypeVar("LogDataType", bound="BaseLogData")


class LoggerProtocol(Protocol):
    """Protocol for logger objects."""

    def log_iteration(self, **kwargs) -> None: ...
    def get_logs(self) -> Any: ...
    def clear_logs(self) -> None: ...


@dataclass
class BaseLogData:
    """Universal diagnostic fields every algorithm produces.

    Subclasses override :py:meth:`clear` and :py:meth:`to_dict` calling
    ``super()`` so the chain handles itself top-down.
    """

    iteration: list[int] = field(default_factory=list)
    evaluations: list[int] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)
    best_solution: list[NDArray[np.float64]] = field(default_factory=list)

    def clear(self) -> None:
        """Reset all log fields. Subclasses extend with ``super().clear()``."""
        self.iteration.clear()
        self.evaluations.clear()
        self.best_fitness.clear()
        self.best_solution.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        """Convert to dict. Subclasses extend with ``super().to_dict() | {...}``."""
        return {
            "iteration": self.iteration,
            "evaluations": self.evaluations,
            "best_fitness": self.best_fitness,
            "best_solution": self.best_solution,
        }


@dataclass
class PopulationLogData(BaseLogData):
    """Adds population-distribution fields used by evolutionary algorithms.

    DES, CMA-ES, and MF-CMA-ES all extend this. L-BFGS-B doesn't — it
    has no population, so inheriting these would just clutter its
    LogData with empty lists.
    """

    worst_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    std_fitness: list[float] = field(default_factory=list)
    population: list[NDArray[np.float64]] = field(default_factory=list)
    eigenvalues: list[NDArray[np.float64]] = field(default_factory=list)
    condition_number: list[float] = field(default_factory=list)

    def clear(self) -> None:
        super().clear()
        self.worst_fitness.clear()
        self.mean_fitness.clear()
        self.std_fitness.clear()
        self.population.clear()
        self.eigenvalues.clear()
        self.condition_number.clear()

    def to_dict(self) -> dict[str, list[Any]]:
        return super().to_dict() | {
            "worst_fitness": self.worst_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "population": self.population,
            "eigenvalues": self.eigenvalues,
            "condition_number": self.condition_number,
        }


class BaseLogger(ABC, Generic[LogDataType]):
    """Base class for algorithm-specific loggers."""

    def __init__(
        self, config, algorithm: AlgorithmChoice = AlgorithmChoice.Unknown
    ) -> None:
        self.config = config
        self.algorithm = algorithm
        self.logs: LogDataType = self._create_log_data()

    @abstractmethod
    def _create_log_data(self) -> LogDataType:
        """Create algorithm-specific log data container."""
        pass

    @abstractmethod
    def log_iteration(self, **kwargs) -> None:
        """Log iteration data."""
        pass

    def get_logs(self) -> LogDataType:
        """Get all logged data."""
        return self.logs

    def get_logs_dict(self) -> dict[str, Any]:
        """Get all logged data as dictionary."""
        return self.logs.to_dict()

    def clear_logs(self) -> None:
        """Clear all logged data."""
        self.logs.clear()

    def get_algorithm(self) -> AlgorithmChoice:
        """Get the algorithm name."""
        return self.algorithm
