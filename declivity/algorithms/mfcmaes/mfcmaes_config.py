from dataclasses import dataclass, field
import numpy as np
import math
from numpy.typing import NDArray

from declivity.core.config_base import PopulationBaseConfig


def default_population_size(dimensions: int) -> int:
    """Default population size based on dimensions.

    Matches the R reference (``nm-cma-es-vectorized.R`` line 48):
    ``lambda = 4*N``.  This differs from the textbook Hansen recipe
    ``4 + floor(3*log(N))`` — the R reference's larger default is what
    the MF-CMA-ES paper validates on.
    """
    return 4 * dimensions


def default_mu(population_size: int) -> int:
    """Default number of parents based on population size."""
    return population_size // 2


def default_window_size(dimensions: int) -> int:
    """Default window size for archive: h = 20 + 1.4*n"""
    return math.floor(20 + 1.4 * dimensions)


def compute_weights(population_size: int, mu: int) -> tuple[NDArray[np.float64], float]:
    """Compute recombination weights for MF-CMA-ES.

    Matches the R reference (``nm-cma-es-vectorized.R`` line 51):
    ``weights = rep(1, mu)`` normalised to sum to one, giving uniform
    ``1/mu`` weights and ``mu_eff = mu``.  The textbook log-decreasing
    weights ``log(mu + 0.5) - log(i + 1)`` are *not* used here — the
    R reference deliberately uses uniform weighting and several of the
    downstream constants (``cc``, ``damps``) are calibrated for that
    choice.
    """
    weights = np.ones(mu, dtype=np.float64) / mu
    mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)
    return weights, mu_eff


@dataclass
class MFCMAESConfig(PopulationBaseConfig):
    """
    Configuration for the Matrix-Free CMA-ES optimizer.
    Extends PopulationBaseConfig with MF-CMA-ES-specific parameters.
    """

    sigma: float = 1.0
    """Initial step size (standard deviation)"""

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

    cs: float = field(init=False)
    """Damping-related learning rate (``(mueff+2)/(N+mueff+3)`` in R)."""

    c_cov: float = field(init=False)
    """Learning rate for covariance matrix adaptation"""

    c_1: float = field(init=False)
    """Learning rate for rank-one update component"""

    c_mu: float = field(init=False)
    """Learning rate for rank-mu update component"""

    damps: float = field(init=False)
    """Damping factor used by the flatland-escape sigma bump."""

    do_flatland_escape: bool = True
    """Whether to bump ``sigma`` when the population collapses onto a
    flat fitness plateau.  Matches R's ``do_flatland_escape`` default."""

    def __post_init__(self) -> None:
        """Calculate derived parameters that depend on other params"""
        if self.population_size <= 0:
            self.population_size = default_population_size(self.dimensions)
        if self.window <= 0:
            self.window = default_window_size(self.dimensions)

        self._recalculate_derived_params()

    def _recalculate_derived_params(self) -> None:
        """Recalculate derived parameters using the R reference formulas
        (``nm-cma-es-vectorized.R`` lines 54–68)."""
        self.tolx = 1e-12 * self.sigma

        self.mu = default_mu(self.population_size)

        # Compute weights and mu_eff (uniform weights, mu_eff == mu).
        self.weights, self.mu_eff = compute_weights(self.population_size, self.mu)

        n_dim = self.dimensions

        # Path cumulation rate — R: ``cc = 4/(N+4)``.
        self.cs = (self.mu_eff + 2.0) / (n_dim + self.mu_eff + 3.0)
        self.cc = 4.0 / (n_dim + 4.0)

        # Covariance update split — R: derive ``ccov`` first, then split
        # into rank-1 and rank-mu shares.  ``mucov`` is just ``mu_eff``.
        mucov = self.mu_eff
        c_cov = (
            (1.0 / mucov) * 2.0 / (n_dim + 1.4) ** 2
            + (1.0 - 1.0 / mucov)
            * ((2.0 * mucov - 1.0) / ((n_dim + 2.0) ** 2 + 2.0 * mucov))
        )
        self.c_cov = c_cov
        self.c_mu = c_cov * (1.0 - 1.0 / mucov)
        self.c_1 = c_cov - self.c_mu

        # Damping factor (used by flatland-escape only).
        self.damps = 1.0 + 2.0 * max(
            0.0, math.sqrt((self.mu_eff - 1.0) / (n_dim + 1.0)) - 1.0
        ) + self.cs

        self.validate()

    def __setattr__(self, name: str, value) -> None:
        """Override setattr to recalculate derived params when key params change."""
        super().__setattr__(name, value)

        if name in ("population_size", "window") and hasattr(self, "mu"):
            self._recalculate_derived_params()

    # MF-CMA-ES has no algorithm-specific diag flags — sigma, p_succ,
    # midpoint_fitness, constraint_violations, evolution path, and mean
    # vector are all logged unconditionally by MFCMAESLogger. The base
    # ``enable_all_diagnostics`` (diag_pop, diag_eigen) is sufficient.

    def __str__(self) -> str:
        return (
            f"MFCMAESConfig(dimensions={self.dimensions}, "
            f"population_size={self.population_size}, window={self.window}, "
            f"sigma={self.sigma}, mu={self.mu}, mu_eff={self.mu_eff:.2f})"
        )
