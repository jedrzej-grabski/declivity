from dataclasses import dataclass, field
import numpy as np
import math
from numpy.typing import NDArray

from src.core.config_base import BaseConfig


def default_population_size(dimensions: int) -> int:
    """Default population size based on dimensions."""
    return 4 + math.floor(3 * math.log(dimensions))


def default_budget(dimensions: int) -> int:
    """Default budget based on dimensions."""
    return 10000 * dimensions


def default_mu(population_size: int) -> int:
    """Default number of parents based on population size."""
    return population_size // 2


def default_window_size(dimensions: int) -> int:
    """Default window size for archive: h = 20 + 1.4*n"""
    return math.floor(20 + 1.4 * dimensions)


def compute_weights(population_size: int, mu: int) -> tuple[NDArray[np.float64], float]:
    """
    Compute recombination weights for MF-CMA-ES.
    Returns: (weights, mu_eff)
    """
    # Create weights_prime for mu individuals
    weights_prime = np.array(
        [max(0, math.log(mu + 0.5) - math.log(i + 1)) for i in range(mu)]
    )

    # Normalize to sum to 1
    weights = weights_prime / np.sum(weights_prime)

    # Calculate mu_eff
    mu_eff = 1.0 / np.sum(weights**2)

    return weights, mu_eff


@dataclass
class MFCMAESConfig(BaseConfig):
    """
    Configuration for the Matrix-Free CMA-ES optimizer.
    Extends BaseConfig with MF-CMA-ES-specific parameters.
    """

    sigma: float = 1.0
    """Initial step size (standard deviation)"""

    population_size: int = field(default=0)
    """Size of the population (0 means use default)"""

    budget: int = field(default=0)
    """Maximum number of function evaluations (0 means use default)"""

    window: int = field(default=0)
    """Archive window size h (0 means use default: 20 + 1.4*n)"""

    # Step-size adaptation (PPMF) parameters
    use_ppmf: bool = True
    """Enable/disable PPMF step-size adaptation (if False, sigma remains constant)"""

    p_target_ppmf: float = 0.1
    """Target success probability for PPMF (default 0.1 from R code, paper suggests 0.2)"""

    damps_ppmf: float = 2.0
    """Damping parameter for PPMF step-size adaptation"""

    # Termination criteria
    tolfun: float = 1e-12
    """Tolerance for function value differences"""

    tolx: float = field(init=False)
    """Tolerance for changes in x"""

    tolxup: float = 1e4
    """Upper tolerance for step size"""

    # Computed/derived parameters
    mu: int = field(init=False)
    """Number of parents"""

    weights: NDArray[np.float64] = field(init=False)
    """Recombination weights"""

    mu_eff: float = field(init=False)
    """Variance effectiveness of the sum of weighted updates"""

    cc: float = field(init=False)
    """Learning rate for cumulation for the rank-one update"""

    c_cov: float = field(init=False)
    """Learning rate for covariance matrix adaptation"""

    c_1: float = field(init=False)
    """Learning rate for rank-one update component"""

    c_mu: float = field(init=False)
    """Learning rate for rank-mu update component"""

    maxit: int = field(init=False)
    """Maximum iterations"""

    def __post_init__(self) -> None:
        """Calculate derived parameters that depend on other params"""
        if self.budget <= 0:
            self.budget = default_budget(self.dimensions)
        if self.population_size <= 0:
            self.population_size = default_population_size(self.dimensions)
        if self.window <= 0:
            self.window = default_window_size(self.dimensions)

        self._recalculate_derived_params()

    def _recalculate_derived_params(self) -> None:
        """Recalculate all derived parameters based on current population_size."""
        self.tolx = 1e-12 * self.sigma

        self.mu = default_mu(self.population_size)

        # Compute weights and mu_eff
        self.weights, self.mu_eff = compute_weights(self.population_size, self.mu)

        n_dim = self.dimensions

        # Learning rate for cumulation for the rank-one update
        self.cc = (4 + self.mu_eff / n_dim) / (n_dim + 4 + 2 * self.mu_eff / n_dim)

        # Covariance matrix adaptation learning rates
        alpha_cov = 2.0
        self.c_1 = alpha_cov / ((n_dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            alpha_cov
            * (self.mu_eff - 2 + 1 / self.mu_eff)
            / ((n_dim + 2) ** 2 + alpha_cov * self.mu_eff / 2),
        )

        # Total covariance adaptation rate
        self.c_cov = self.c_1 + self.c_mu

        self.maxit = math.floor(self.budget / self.population_size)

        self.validate()

    def __setattr__(self, name: str, value) -> None:
        """Override setattr to recalculate derived params when key params change."""
        super().__setattr__(name, value)

        if name in ("population_size", "budget", "window") and hasattr(self, "mu"):
            self._recalculate_derived_params()

    # MF-CMA-ES has no algorithm-specific diag flags — sigma, p_succ,
    # midpoint_fitness, constraint_violations, evolution path, and mean
    # vector are all logged unconditionally by MFCMAESLogger. The base
    # ``enable_all_diagnostics`` (diag_pop, diag_eigen) is sufficient.

    def __str__(self) -> str:
        return (
            f"MFCMAESConfig(dimensions={self.dimensions}, budget={self.budget}, "
            f"population_size={self.population_size}, window={self.window}, "
            f"sigma={self.sigma}, mu={self.mu}, mu_eff={self.mu_eff:.2f})"
        )
