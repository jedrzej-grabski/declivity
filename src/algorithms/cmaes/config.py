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


def compute_weights(
    population_size: int, mu: int, dimensions: int
) -> tuple[NDArray[np.float64], float, float, float, float]:
    """
    Compute recombination weights and related parameters.
    Returns: (weights, mu_eff, c1, cmu, min_alpha)
    """
    lambda_ = population_size

    # Create weights_prime for all lambda individuals (correct formula from reference)
    weights_prime = np.array(
        [max(0, math.log(mu + 0.5) - math.log(i + 1)) for i in range(mu)]
    )

    # Pad with zeros for individuals beyond mu
    weights_prime = np.concatenate([weights_prime, np.zeros(lambda_ - mu)])

    # Normalize positive weights to sum to 1
    weights_prime[:mu] = weights_prime[:mu] / np.sum(weights_prime[:mu])

    # Calculate mu_eff from normalized positive weights
    mu_eff = 1.0 / np.sum(weights_prime[:mu] ** 2)

    # Learning rates
    n_dim = dimensions
    c1 = 2.0 / ((n_dim + 1.3) ** 2 + mu_eff)
    cmu = min(
        1 - c1,
        2.0 * (mu_eff - 2 + 1 / mu_eff) / ((n_dim + 2) ** 2 + mu_eff),
    )

    # For negative weights (active CMA-ES), we'll handle them in the optimizer
    min_alpha = 1.0  # Not used in basic implementation

    return weights_prime, mu_eff, c1, cmu, min_alpha


def compute_maxit(budget: int, population_size: int) -> int:
    """Compute maximum iterations based on budget and population size."""
    return math.floor(budget / population_size)


@dataclass
class CMAESConfig(BaseConfig):
    """
    Configuration for the CMA-ES optimizer.
    Extends BaseConfig with CMA-ES-specific parameters.
    """

    sigma: float = 0.0
    """Initial step size (standard deviation) - 0 means auto-calculate"""

    population_size: int = field(default=0)
    """Size of the population (0 means use default)"""

    budget: int = field(default=0)
    """Maximum number of function evaluations (0 means use default)"""

    # CMA-ES-specific diagnostic logging
    diag_sigma: bool = False
    """Log sigma values each generation."""

    diag_covariance_matrix: bool = False
    """Store the full covariance matrix at every generation. Useful for
    tracking how the covariance evolves, but expensive in memory for
    long runs (n^2 floats per generation)."""

    # Termination criteria
    tolfun: float = 1e-12
    """Tolerance for function value differences"""

    tolx: float = field(init=False)
    """Tolerance for changes in x"""

    tolxup: float = 1e4
    """Upper tolerance for step size"""

    tolconditioncov: float = 1e14
    """Tolerance for condition number of covariance matrix"""

    # Computed/derived parameters (will be calculated in __post_init__)
    mu: int = field(init=False)
    """Number of parents"""

    weights: NDArray[np.float64] = field(init=False)
    """Recombination weights"""

    mu_eff: float = field(init=False)
    """Variance effectiveness of the sum of weighted updates"""

    cc: float = field(init=False)
    """Learning rate for cumulation for the rank-one update"""

    c_sigma: float = field(init=False)
    """Learning rate for cumulation for the step-size control"""

    c1: float = field(init=False)
    """Learning rate for the rank-one update"""

    cmu: float = field(init=False)
    """Learning rate for the rank-μ update"""

    d_sigma: float = field(init=False)
    """Damping parameter for step-size"""

    chi_n: float = field(init=False)
    """Expected length of a standard normal random vector"""

    cm: float = 1.0
    """Step size multiplier for mean update"""

    maxit: int = field(init=False)
    """Maximum iterations"""

    funhist_term: int = field(init=False)
    """Number of generations for function value history"""

    def __post_init__(self) -> None:
        """Calculate derived parameters that depend on other params"""
        if self.budget <= 0:
            self.budget = default_budget(self.dimensions)
        if self.population_size <= 0:
            self.population_size = default_population_size(self.dimensions)

        self._recalculate_derived_params()

    def _recalculate_derived_params(self) -> None:
        """Recalculate all derived parameters based on current population_size."""
        self.tolx = 1e-12 * self.sigma

        self.mu = default_mu(self.population_size)

        # Compute weights and learning rates together
        self.weights, self.mu_eff, self.c1, self.cmu, _ = compute_weights(
            self.population_size, self.mu, self.dimensions
        )

        n_dim = self.dimensions

        # Learning rate for cumulation for the rank-one update
        self.cc = (4 + self.mu_eff / n_dim) / (n_dim + 4 + 2 * self.mu_eff / n_dim)

        # Learning rate for cumulation for the step-size control
        self.c_sigma = (self.mu_eff + 2) / (n_dim + self.mu_eff + 5)

        # Damping parameter for step-size
        self.d_sigma = (
            1
            + 2 * max(0, math.sqrt((self.mu_eff - 1) / (n_dim + 1)) - 1)
            + self.c_sigma
        )

        # Expected length of standard normal random vector
        self.chi_n = math.sqrt(n_dim) * (
            1.0 - (1.0 / (4.0 * n_dim)) + 1.0 / (21.0 * (n_dim**2))
        )

        self.maxit = compute_maxit(self.budget, self.population_size)
        self.funhist_term = 10 + math.ceil(30 * n_dim / self.population_size)

        self.validate()

    def __setattr__(self, name: str, value) -> None:
        """Override setattr to recalculate derived params when population_size changes."""
        super().__setattr__(name, value)

        # Recalculate derived params when population_size or budget changes
        # Only do this after __post_init__ has run (check if mu exists)
        if name in ("population_size", "budget") and hasattr(self, "mu"):
            self._recalculate_derived_params()

    def enable_all_diagnostics(self) -> None:
        super().enable_all_diagnostics()
        self.diag_sigma = True
        self.diag_covariance_matrix = True

    def __str__(self) -> str:
        return (
            f"CMAESConfig(dimensions={self.dimensions}, budget={self.budget}, "
            f"population_size={self.population_size}, sigma={self.sigma}, "
            f"mu={self.mu}, mu_eff={self.mu_eff:.2f})"
        )
