"""Configuration for the framework-native CMA-ES implementation.

The math here mirrors Hansen's active-CMA-ES (eq. 49–56, p. 28) and
matches the precomputed constants used by
:class:`~declivity.algorithms.cmaes.cmaes_reference.CMA`, so the two
implementations stay numerically aligned on the cross-validation
experiments under ``experiments/cross_validation/``.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from declivity.core.config_base import PopulationBaseConfig


def default_population_size(dimensions: int) -> int:
    """Default population size (eq. 48)."""
    return 4 + math.floor(3 * math.log(dimensions))


def compute_weights_and_rates(
    population_size: int, dimensions: int
) -> tuple[NDArray[np.float64], int, float, float, float, float, float, float]:
    """Compute the active-CMA weights and the matching learning rates.

    Returns ``(weights, mu, mu_eff, c1, cmu, c_sigma, d_sigma, cc)`` so that
    callers can fill out the config dataclass in one pass.  The formulas
    follow Hansen 2016 eqs. 49–56 and reproduce the reference
    implementation in :mod:`declivity.algorithms.cmaes.cmaes_reference`.
    """
    n_dim = dimensions
    lambda_ = population_size
    mu = lambda_ // 2

    # eq. 49 — preliminary weights for *all* λ individuals, half positive,
    # half negative (active CMA-ES).
    weights_prime = np.array(
        [math.log((lambda_ + 1) / 2) - math.log(i + 1) for i in range(lambda_)]
    )
    mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)
    mu_eff_minus = (np.sum(weights_prime[mu:]) ** 2) / np.sum(weights_prime[mu:] ** 2)

    # Learning rates for the rank-one and rank-μ updates (Hansen 2016, p. 27).
    alpha_cov = 2.0
    c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
    cmu = min(
        1.0 - c1 - 1e-8,
        alpha_cov
        * (mu_eff - 2.0 + 1.0 / mu_eff)
        / ((n_dim + 2.0) ** 2 + alpha_cov * mu_eff / 2.0),
    )

    # eqs. 50–52 — bound for the negative-weight scaling factor.
    min_alpha = min(
        1.0 + c1 / cmu,
        1.0 + (2.0 * mu_eff_minus) / (mu_eff + 2.0),
        (1.0 - c1 - cmu) / (n_dim * cmu),
    )

    # eq. 53 — final active weights with positive sum normalised to 1 and
    # negative tail scaled by ``min_alpha / |negative sum|``.
    positive_sum = float(np.sum(weights_prime[weights_prime > 0]))
    negative_sum = float(np.sum(np.abs(weights_prime[weights_prime < 0])))
    weights = np.where(
        weights_prime >= 0,
        weights_prime / positive_sum,
        (min_alpha / negative_sum) * weights_prime,
    )

    # eq. 55–56 — cumulation rates.
    c_sigma = (mu_eff + 2.0) / (n_dim + mu_eff + 5.0)
    d_sigma = (
        1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n_dim + 1.0)) - 1.0) + c_sigma
    )
    cc = (4.0 + mu_eff / n_dim) / (n_dim + 4.0 + 2.0 * mu_eff / n_dim)

    return weights, mu, mu_eff, c1, cmu, c_sigma, d_sigma, cc


def chi_n_expected(dimensions: int) -> float:
    """E‖N(0, I_d)‖ (Hansen 2016, p. 28)."""
    n_dim = dimensions
    return math.sqrt(n_dim) * (1.0 - 1.0 / (4.0 * n_dim) + 1.0 / (21.0 * n_dim**2))


@dataclass
class CMAESConfig(PopulationBaseConfig):
    """Configuration for the CMA-ES optimizer."""

    sigma: float = 0.0
    """Initial step size; ``0.0`` means auto-set from the bounds range."""

    population_size: int = field(default=0)
    """Population size λ (``0`` → default ``4 + ⌊3 ln d⌋``)."""

    diag_covariance_matrix: bool = False
    """Store the full covariance matrix every generation (memory-heavy)."""

    # Termination — defaults match the reference implementation.
    tolfun: float = 1e-12
    tolxup: float = 1e4
    tolconditioncov: float = 1e14

    tolx: float = field(init=False)
    """``1e-12 · sigma`` (rescaled in ``__post_init__``)."""

    cm: float = 1.0
    """Mean-update damping (eq. 54)."""

    # Derived constants — populated by ``_recalculate_derived_params``.
    mu: int = field(init=False)
    weights: NDArray[np.float64] = field(init=False)
    mu_eff: float = field(init=False)
    c1: float = field(init=False)
    cmu: float = field(init=False)
    c_sigma: float = field(init=False)
    d_sigma: float = field(init=False)
    cc: float = field(init=False)
    chi_n: float = field(init=False)
    funhist_term: int = field(init=False)

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            self.population_size = default_population_size(self.dimensions)
        self._recalculate_derived_params()

    def _recalculate_derived_params(self) -> None:
        (
            self.weights,
            self.mu,
            self.mu_eff,
            self.c1,
            self.cmu,
            self.c_sigma,
            self.d_sigma,
            self.cc,
        ) = compute_weights_and_rates(self.population_size, self.dimensions)
        self.chi_n = chi_n_expected(self.dimensions)
        self.tolx = 1e-12 * self.sigma
        self.funhist_term = 10 + math.ceil(30 * self.dimensions / self.population_size)
        self.validate()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in ("population_size", "sigma", "dimensions") and "mu" in self.__dict__:
            self._recalculate_derived_params()

    def enable_all_diagnostics(self) -> None:
        super().enable_all_diagnostics()
        self.diag_covariance_matrix = True

    def __str__(self) -> str:
        return (
            f"CMAESConfig(dimensions={self.dimensions}, "
            f"population_size={self.population_size}, sigma={self.sigma}, "
            f"mu={self.mu}, mu_eff={self.mu_eff:.2f})"
        )
