"""Base configuration dataclasses for every optimizer.

Two-level hierarchy:

* :class:`BaseConfig` — the minimum every optimizer (single-point or
  population-based) needs: ``dimensions`` and ``budget``.
* :class:`PopulationBaseConfig` — adds the fields and diagnostic flags
  that only matter to population-based algorithms (``population_size``,
  ``diag_pop``, ``diag_eigen``).

L-BFGS-B inherits directly from :class:`BaseConfig`.  DES, CMA-ES, and
MF-CMA-ES inherit from :class:`PopulationBaseConfig`.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


class ConfigProtocol(Protocol):
    """Protocol every optimizer configuration satisfies."""

    dimensions: int
    budget: int

    def validate(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class BaseConfig:
    """Root configuration shared by every optimizer.

    Carries only what *every* optimizer (single-point or
    population-based) needs.  Population-only fields live on
    :class:`PopulationBaseConfig`.
    """

    dimensions: int
    """Number of dimensions in the problem"""

    budget: int = field(default=0)
    """Maximum number of function evaluations"""

    def validate(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be a positive integer.")
        if self.budget <= 0:
            raise ValueError("Budget must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "budget": self.budget,
        }

    def enable_all_diagnostics(self) -> None:
        """No-op base hook.  Subclasses override to enable their flags;
        chain with ``super().enable_all_diagnostics()`` so the whole
        diagnostic surface flips on with a single call."""
        pass

    def disable_all_diagnostics(self) -> None:
        """No-op base hook — symmetric counterpart to
        :meth:`enable_all_diagnostics`."""
        pass


@dataclass
class PopulationBaseConfig(BaseConfig):
    """Configuration shared by every population-based optimizer.

    Adds the population-size field and the two diagnostic flags that
    only population-based algorithms gate on:

    - ``diag_pop`` — store the full population at every iteration. Off
      by default because the memory cost is
      ``pop_size * dim * sizeof(float)`` per iteration; only enable
      when you need to inspect or replot population clouds.
    - ``diag_eigen`` — compute and log eigenvalues / condition number
      of the population covariance (DES, CMA-ES, MF-CMA-ES). Off by
      default because the eigendecomposition adds per-iteration cost;
      enable when tracking covariance geometry.
    """

    population_size: int = field(default=0)
    """Size of the population"""

    diag_pop: bool = False
    """Log full populations each iteration (memory-expensive)."""

    diag_eigen: bool = False
    """Log eigenvalues / condition number of the population covariance."""

    def validate(self) -> None:
        super().validate()
        if self.population_size <= 0:
            raise ValueError("Population size must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "population_size": self.population_size,
            "diag_pop": self.diag_pop,
            "diag_eigen": self.diag_eigen,
        }

    def enable_all_diagnostics(self) -> None:
        super().enable_all_diagnostics()
        self.diag_pop = True
        self.diag_eigen = True

    def disable_all_diagnostics(self) -> None:
        super().disable_all_diagnostics()
        self.diag_pop = False
        self.diag_eigen = False
