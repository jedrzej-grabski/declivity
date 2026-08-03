"""
Population initializers for evolutionary algorithms.

Each concrete class encapsulates how a population is seeded at the start
of a run, making initialization pluggable without touching algorithm logic.

Hierarchy
---------
- ``PopulationInitializer`` — abstract base (single abstract method)
- ``NormalPopulationInitializer`` — DES default; matches ``rng.normal(x0, (ub-lb)/scale_factor)``
- ``MeanSigmaPopulationInitializer`` — MF-CMA-ES default; dim-first RNG layout preserved
- ``IdentityPopulationInitializer`` — explicit no-op placeholder (raises NotImplementedError)
- ``PopulationInitializerType`` — discoverability enum with ``.build()`` factory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class PopulationInitializer(ABC):
    """Abstract base class for population initialization strategies.

    Subclasses implement :meth:`generate_population` to produce an initial
    population matrix of shape ``(pop_size, dim)`` for a given starting
    point, bounds, and random state.
    """

    @abstractmethod
    def generate_population(
        self,
        rng: np.random.Generator,
        x0: NDArray[np.float64],
        pop_size: int,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Generate an initial population.

        Parameters
        ----------
        rng:
            NumPy random generator (caller-owned; this method may advance it).
        x0:
            Starting point / mean, shape ``(dim,)``.
        pop_size:
            Number of individuals to generate.
        lower_bounds:
            Per-dimension lower bounds, shape ``(dim,)``.
        upper_bounds:
            Per-dimension upper bounds, shape ``(dim,)``.

        Returns
        -------
        NDArray[np.float64]
            Population matrix of shape ``(pop_size, dim)``.
        """
        ...


class NormalPopulationInitializer(PopulationInitializer):
    """DES-default initializer — normally distributed around *x0*.

    Reproduces DES's inline code::

        sigma = (ub - lb) / 6
        population = rng.normal(loc=x0, scale=sigma, size=(pop_size, dim))

    The ``scale_factor`` parameter controls the number of standard deviations
    that span the search range; the DES default is ``6`` so that ~99.7 % of
    the initial samples fall inside ``[lb, ub]``.
    """

    def __init__(self, scale_factor: float = 6.0) -> None:
        self.scale_factor = scale_factor

    def generate_population(
        self,
        rng: np.random.Generator,
        x0: NDArray[np.float64],
        pop_size: int,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        scale = (upper_bounds - lower_bounds) / self.scale_factor
        return rng.normal(loc=x0, scale=scale, size=(pop_size, len(x0)))


class MeanSigmaPopulationInitializer(PopulationInitializer):
    """MF-CMA-ES-default initializer — isotropic Gaussian scaled by *sigma*.

    Reproduces MF-CMA-ES's inline code::

        initial_d = rng.standard_normal((dim, pop_size))   # dim-first layout
        initial_arx = mean[:, np.newaxis] + sigma * initial_d

    The population is returned transposed to ``(pop_size, dim)``; the
    dim-first sampling order is preserved so that downstream code that
    recovers direction vectors via ``d = (arx - mean) / sigma`` gets
    numerically identical results.

    Parameters
    ----------
    sigma:
        Step-size scaling for the initial population.  Should match
        ``MFCMAESConfig.sigma`` (default ``1.0``).
    """

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def generate_population(
        self,
        rng: np.random.Generator,
        x0: NDArray[np.float64],
        pop_size: int,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Generate in (dim, pop_size) layout to match MF-CMA-ES's RNG sequence.
        d = rng.standard_normal((len(x0), pop_size))
        arx = x0[:, np.newaxis] + self.sigma * d
        return arx.T  # (pop_size, dim)


class UniformPopulationInitializer(PopulationInitializer):
    """R-DES-default initializer — uniform per-individual inside a fraction of bounds.

    Reproduces the R reference's inline draw::

        replicate(lambda, runif(N, fraction*lower, fraction*upper))

    The starting point ``x0`` is deliberately **not** used: R-DES does
    not centre the first population on ``par``.  The default
    ``fraction=0.8`` matches ``DES.R`` line 202.
    """

    def __init__(self, fraction: float = 0.8) -> None:
        self.fraction = fraction

    def generate_population(
        self,
        rng: np.random.Generator,
        x0: NDArray[np.float64],
        pop_size: int,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del x0  # intentionally unused — R-DES draws independently of par
        low = self.fraction * lower_bounds
        high = self.fraction * upper_bounds
        # R's ``replicate(lambda, runif(N, low, high))`` draws (N, lambda)
        # column-by-column.  Preserve that draw order so a side-by-side
        # NumPy-seeded reference port consumes the same RNG stream.
        out = rng.uniform(
            low=low[:, None], high=high[:, None], size=(len(low), pop_size)
        )
        return out.T  # (pop_size, dim)


class IdentityPopulationInitializer(PopulationInitializer):
    """Explicit no-op placeholder — must not be called.

    Use this when an optimiser generates its own candidates without going
    through a :class:`PopulationInitializer` and the framework still
    requires one for structural enforcement.  Currently no shipped
    algorithm uses it as a default (CMA-ES, DES, and MF-CMA-ES all route
    iteration-0 sampling through a real initializer), but it remains
    available for custom optimisers that need an explicit "this seam is
    intentionally unused" marker.

    Calling :meth:`generate_population` raises :exc:`NotImplementedError`.
    """

    def generate_population(
        self,
        rng: np.random.Generator,
        x0: NDArray[np.float64],
        pop_size: int,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            "IdentityPopulationInitializer is a structural placeholder and "
            "must not be called directly. The optimiser using it should "
            "produce its own candidates without invoking this method."
        )


class PopulationInitializerType(Enum):
    """Discoverability enum for built-in population initializers.

    Each variant maps to a concrete :class:`PopulationInitializer` and
    exposes a :meth:`build` factory that returns a ready-to-use instance
    with sensible defaults.

    Examples
    --------
    >>> initializer = PopulationInitializerType.NORMAL.build()
    >>> population = initializer.generate_population(rng, x0, pop_size, lb, ub)
    """

    NORMAL = "normal"
    MEAN_SIGMA = "mean_sigma"
    UNIFORM = "uniform"
    IDENTITY = "identity"

    def build(self) -> PopulationInitializer:
        """Construct the corresponding :class:`PopulationInitializer` instance."""
        match self:
            case PopulationInitializerType.NORMAL:
                return NormalPopulationInitializer()
            case PopulationInitializerType.MEAN_SIGMA:
                return MeanSigmaPopulationInitializer(sigma=1.0)
            case PopulationInitializerType.UNIFORM:
                return UniformPopulationInitializer()
            case PopulationInitializerType.IDENTITY:
                return IdentityPopulationInitializer()
