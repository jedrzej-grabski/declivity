from dataclasses import dataclass

from declivity.core.config_base import PopulationBaseConfig

__all__ = [
    "NelderMeadConfig",
]


@dataclass
class NelderMeadConfig(PopulationBaseConfig):
    """Configuration for the Nelder-Mead simplex method.

    Nelder-Mead maintains a simplex of ``dimensions + 1`` vertices — a
    population in the framework's sense — so this inherits
    :class:`PopulationBaseConfig` (``diag_pop`` stores the full simplex
    each iteration, ``diag_eigen`` logs the simplex-covariance geometry).
    ``population_size`` is structurally fixed to ``dimensions + 1``; it
    is derived automatically and validated, not chosen.

    The reflection/expansion/contraction/shrink coefficients follow the
    standard values (rho=1, chi=2, psi=0.5, sigma=0.5) or, with
    ``adaptive=True``, the dimension-dependent variant of Gao & Han
    (2012) that behaves better in high dimension.
    """

    xatol: float = 1e-4
    """Internal convergence tolerance on the simplex extent: stop when
    ``max |sim[1:] - sim[0]| <= xatol`` (jointly with ``fatol``)."""

    fatol: float = 1e-4
    """Internal convergence tolerance on the fitness spread: stop when
    ``max |f(sim[0]) - f(sim[1:])| <= fatol`` (jointly with ``xatol``)."""

    adaptive: bool = False
    """Use the Gao-Han (2012) dimension-adaptive coefficients instead of
    the classic constants. Useful for high-dimensional problems."""

    # Diagnostic flags specific to Nelder-Mead
    diag_operations: bool = False
    """Log which simplex operation each iteration performed (reflect /
    expand / contract-outside / contract-inside / shrink)."""

    diag_volume: bool = False
    """Log the simplex volume each iteration (O(n^3) determinant)."""

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            self.population_size = self.dimensions + 1
        self.validate()

    def validate(self) -> None:
        super().validate()
        if self.population_size != self.dimensions + 1:
            raise ValueError(
                "Nelder-Mead's simplex has exactly dimensions + 1 vertices; "
                f"population_size={self.population_size} conflicts with "
                f"dimensions={self.dimensions}."
            )
        if self.xatol <= 0:
            raise ValueError("xatol must be positive.")
        if self.fatol <= 0:
            raise ValueError("fatol must be positive.")

    # Simplex operation coefficients (Gao-Han adaptive or classic).

    @property
    def rho(self) -> float:
        """Reflection coefficient."""
        return 1.0

    @property
    def chi(self) -> float:
        """Expansion coefficient."""
        if self.adaptive:
            return 1.0 + 2.0 / self.dimensions
        return 2.0

    @property
    def psi(self) -> float:
        """Contraction coefficient."""
        if self.adaptive:
            return 0.75 - 1.0 / (2.0 * self.dimensions)
        return 0.5

    @property
    def sigma_shrink(self) -> float:
        """Shrink coefficient (named ``sigma`` in the literature; renamed
        here to avoid colliding with the CMA-ES step size)."""
        if self.adaptive:
            return 1.0 - 1.0 / self.dimensions
        return 0.5

    def enable_all_diagnostics(self) -> None:
        super().enable_all_diagnostics()
        self.diag_operations = True
        self.diag_volume = True

    def disable_all_diagnostics(self) -> None:
        super().disable_all_diagnostics()
        self.diag_operations = False
        self.diag_volume = False

    def __str__(self) -> str:
        return (
            f"NelderMeadConfig(dimensions={self.dimensions}, "
            f"xatol={self.xatol:.1e}, fatol={self.fatol:.1e}, "
            f"adaptive={self.adaptive})"
        )
