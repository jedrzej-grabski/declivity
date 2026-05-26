"""Base configuration dataclass for every optimizer.

Diagnostic flags are kept lean: only flags that an optimizer or logger
actually gates on appear here or on subclasses. Flags that toggle
nothing have been removed (they were legacy from an earlier design).
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


class ConfigProtocol(Protocol):
    """Protocol defining the interface for algorithm configurations."""

    dimensions: int
    budget: int
    population_size: int

    def validate(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class BaseConfig:
    """Base configuration class for optimization algorithms.

    Shared parameters every optimizer reads (``dimensions``, ``budget``,
    ``population_size``) plus two diagnostic flags that the
    population-based optimizers gate on:

    - ``diag_pop`` — store the full population at every iteration. Off by
      default because the memory cost is ``pop_size * dim * sizeof(float)``
      per iteration; only enable when you need to inspect or replot
      population clouds.
    - ``diag_eigen`` — compute and log eigenvalues / condition number of
      the population covariance (DES, CMA-ES, MF-CMA-ES). Off by default
      because the eigendecomposition adds per-iteration cost; enable when
      tracking covariance geometry.

    Best-fitness logging is always on — the convergence trace is the
    minimum any benchmark consumer needs and the logging is cheap.
    """

    dimensions: int
    """Number of dimensions in the problem"""

    budget: int = field(default=0)
    """Maximum number of function evaluations"""

    population_size: int = field(default=0)
    """Size of the population (1 for single-point methods)"""

    diag_pop: bool = False
    """Log full populations each iteration (memory-expensive)."""

    diag_eigen: bool = False
    """Log eigenvalues / condition number of the population covariance."""

    def validate(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be a positive integer.")
        if self.budget <= 0:
            raise ValueError("Budget must be a positive integer.")
        if self.population_size <= 0:
            raise ValueError("Population size must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "budget": self.budget,
            "population_size": self.population_size,
            "diag_pop": self.diag_pop,
            "diag_eigen": self.diag_eigen,
        }

    def enable_all_diagnostics(self) -> None:
        """Enable all diagnostic logging options. Subclasses extend via ``super()``."""
        self.diag_pop = True
        self.diag_eigen = True

    def disable_all_diagnostics(self) -> None:
        """Disable all diagnostic logging options. Subclasses extend via ``super()``."""
        self.diag_pop = False
        self.diag_eigen = False
