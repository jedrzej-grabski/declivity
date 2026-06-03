from dataclasses import dataclass, field
import numpy as np
import math
from numpy.typing import NDArray

from src.core.config_base import BaseConfig


def default_population_size(obj: "DESConfig") -> int:
    """Default population size based on dimensions."""
    return 4 * obj.dimensions


def default_budget(obj: "DESConfig") -> int:
    """Default budget based on dimensions."""
    return 10000 * obj.dimensions


def default_cp(obj: "DESConfig") -> float:
    """Default evolution path decay factor based on dimensions."""
    return 1 / np.sqrt(obj.dimensions)


def default_history(obj: "DESConfig") -> int:
    """Default history size based on dimensions."""
    return math.ceil(6 + math.ceil(3 * np.sqrt(obj.dimensions)))


def default_mu(obj: "DESConfig") -> int:
    """Default number of parents based on population size."""
    return math.floor(obj.population_size / 2)


def default_weights(obj: "DESConfig") -> NDArray[np.float64]:
    """Default recombination weights based on mu."""
    weights = np.log(obj.mu + 1) - np.log(np.arange(1, obj.mu + 1))
    return weights / np.sum(weights)


def default_weights_pop(obj: "DESConfig") -> NDArray[np.float64]:
    """Default per-individual weights for ``pop_mean`` — log-decreasing
    in column index, matching DES.R's ``weightsPop`` (line 193)."""
    weights = np.log(obj.population_size + 1) - np.log(
        np.arange(1, obj.population_size + 1)
    )
    return weights / np.sum(weights)


def default_ccum(obj: "DESConfig") -> float:
    """Default cumulation factor based on mu."""
    return obj.mu / (obj.mu + 2)


def default_pathratio(obj: "DESConfig") -> float:
    """Default path ratio based on path length."""
    return np.sqrt(obj.path_length)


def compute_maxit(budget: int, population_size: int) -> int:
    """Compute maximum iterations based on budget and population size."""
    return math.floor(budget / (population_size + 1))


def default_ft_scale(obj: "DESConfig") -> float:
    """Default Ft scaling factor."""
    N = obj.dimensions
    mu_eff = obj.mu_eff
    return ((mu_eff + 2) / (N + mu_eff + 3)) / (
        1
        + 2 * max(0, np.sqrt((mu_eff - 1) / (N + 1)) - 1)
        + (mu_eff + 2) / (N + mu_eff + 3)
    )


@dataclass
class DESConfig(BaseConfig):
    """
    Configuration for the DES optimizer.
    Extends BaseConfig with DES-specific parameters.
    """

    ft: float = 1.0
    """Scaling factor of difference vectors"""

    init_ft: float = 1.0
    """Initial scaling factor"""

    path_length: int = 6
    """Size of evolution path"""

    c_ft: float = 0.0
    """Control parameter for Ft adaptation.

    Defaults to ``0.0`` to match the R reference (``DES.R`` line 65,
    ``controlParam("c_Ft", 0)``).  When zero, :func:`calculate_ft`
    returns ``current_ft`` unchanged, so ``Ft`` remains fixed at
    :attr:`init_ft` for the duration of the run — the same behaviour
    as R-DES, where the ``calculateFt`` call is commented out
    (``DES.R`` lines 271–273).  Set a positive value to enable
    cumulative-step-size-style Ft adaptation.
    """

    tol: float = 1e-12
    """Numerical perturbation scale for the auxiliary noise term in the
    new-population formula (``DES.R`` line 299).  Must stay small
    (default ``1e-12``); larger values inject substantial Gaussian
    noise every iteration and destabilise late-stage convergence."""

    lamarckian: bool = False
    """Whether to use Lamarckian evolution"""

    # DES-specific diagnostic logging
    diag_ft: bool = False
    """Log Ft values"""

    # Computed/derived parameters
    cp: float = field(init=False)
    """Evolution path decay factor"""

    history: int = field(init=False)
    """Size of history window"""

    mu: int = field(init=False)
    """Number of parents"""

    weights: NDArray[np.float64] = field(init=False)
    """Recombination weights"""

    weights_pop: NDArray[np.float64] = field(init=False)
    """Per-individual weights for the log-weighted ``pop_mean``."""

    c_cum: float = field(init=False)
    """Evolution path decay factor"""

    path_ratio: float = field(init=False)
    """Path length control reference value"""

    maxit: int = field(init=False)
    """Maximum iterations"""

    mu_eff: float = field(init=False)
    """Effective selection mass"""

    ft_scale: float = field(init=False)
    """Scaling factor for Ft"""

    def __post_init__(self) -> None:
        """Calculate derived parameters that depend on other params"""
        if self.budget <= 0:
            self.budget = default_budget(self)
        if self.population_size <= 0:
            self.population_size = default_population_size(self)
        self.cp = default_cp(self)
        self.history = default_history(self)

        self.mu = default_mu(self)
        self.weights = default_weights(self)
        self.weights_pop = default_weights_pop(self)
        self.c_cum = default_ccum(self)
        self.path_ratio = default_pathratio(self)

        weights_sum_square = np.sum(self.weights**2)
        self.mu_eff = np.sum(self.weights) ** 2 / weights_sum_square

        self.ft_scale = default_ft_scale(self)

        self.maxit = compute_maxit(self.budget, self.population_size)

        super().validate()

    def enable_all_diagnostics(self) -> None:
        """Enable all diagnostic logging options including DES-specific ones."""
        super().enable_all_diagnostics()
        self.diag_ft = True

    def __str__(self) -> str:
        """String representation of the DESConfig."""
        return (
            f"DESConfig(dimensions={self.dimensions}, budget={self.budget}, "
            f"population_size={self.population_size}, ft={self.ft}, "
            f"init_ft={self.init_ft}, path_length={self.path_length}, "
            f"c_ft={self.c_ft}, lamarckian={self.lamarckian}, "
            f"diag_ft={self.diag_ft})"
        )
