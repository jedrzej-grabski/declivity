"""
Population initializers for evolutionary algorithms.

Each concrete class encapsulates how a population is seeded at the start
of a run, making initialization pluggable without touching algorithm logic.

Hierarchy
---------
- ``PopulationInitializer`` — abstract base (single abstract method)
- ``NormalPopulationInitializer`` — DES default; matches ``rng.normal(x0, (ub-lb)/scale_factor)``
- ``MeanSigmaPopulationInitializer`` — MF-CMA-ES default; dim-first RNG layout preserved
- ``SimplexPopulationInitializer`` — Nelder-Mead default; deterministic axis-step simplex
- ``CovarianceSimplexInitializer`` — Nelder-Mead simplex shaped by a learned geometry (CMA-ES covariance)
- ``IdentityPopulationInitializer`` — explicit no-op placeholder (raises NotImplementedError)
- ``PopulationInitializerType`` — discoverability enum with ``.build()`` factory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from declivity.utils.initial_geometry import InitialGeometry


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
        out = rng.uniform(low=low[:, None], high=high[:, None], size=(len(low), pop_size))
        return out.T  # (pop_size, dim)


class SimplexPopulationInitializer(PopulationInitializer):
    """Nelder-Mead-default initializer — deterministic axis-step simplex.

    Builds the classic ``(n+1, n)`` starting simplex around *x0* the way
    SciPy's ``method='Nelder-Mead'`` does: vertex 0 is ``x0`` itself and
    vertex ``k+1`` perturbs coordinate ``k`` by 5 % (``nonzdelt``), or by
    the absolute step ``zdelt`` where ``x0[k] == 0``::

        sim[0] = x0
        sim[k + 1][k] = (1 + nonzdelt) * x0[k]      if x0[k] != 0
        sim[k + 1][k] = zdelt                        otherwise

    ``x0`` is clipped into the box first, and vertices that land above an
    upper bound are *reflected* into the interior before clipping — the
    same degeneracy guard SciPy applies, so a clipped simplex never
    collapses onto a bound face.

    Deterministic: the ``rng`` argument is accepted for interface
    compatibility but never consumed.  Swap in a random initializer
    (e.g. :class:`NormalPopulationInitializer`) to study how simplex
    seeding affects Nelder-Mead.
    """

    def __init__(self, nonzdelt: float = 0.05, zdelt: float = 0.00025) -> None:
        self.nonzdelt = nonzdelt
        """Relative perturbation for nonzero coordinates of x0."""
        self.zdelt = zdelt
        """Absolute perturbation for zero coordinates of x0."""

    def generate_population(
        self,
        rng: np.random.Generator,
        x0: NDArray[np.float64],
        pop_size: int,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del rng  # deterministic construction — interface compatibility only
        dim = len(x0)
        if pop_size != dim + 1:
            raise ValueError(
                f"A simplex over {dim} dimensions has exactly {dim + 1} "
                f"vertices; got pop_size={pop_size}."
            )

        x0 = np.clip(np.asarray(x0, dtype=float), lower_bounds, upper_bounds)

        sim = np.empty((dim + 1, dim), dtype=float)
        sim[0] = x0
        for k in range(dim):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + self.nonzdelt) * y[k]
            else:
                y[k] = self.zdelt
            sim[k + 1] = y

        # Degeneracy guard: a vertex pushed past an upper bound is
        # reflected into the interior (2*ub - v), then everything is
        # clipped — plain clipping could collapse the simplex onto a
        # bound face when x0 sits near it.
        with np.errstate(invalid="ignore"):
            mask = sim > upper_bounds
            sim = np.where(mask, 2 * upper_bounds - sim, sim)
        sim = np.clip(sim, lower_bounds, upper_bounds)
        return sim


class CovarianceSimplexInitializer(PopulationInitializer):
    """Nelder-Mead initializer that shapes the simplex from a learned geometry.

    Builds the ``(n+1, n)`` starting simplex around *x0* with edges along the
    geometry's principal axes (the eigenvectors of a CMA-ES covariance), so the
    simplex is elongated along flat / high-variance directions and compressed
    along steep ones — matching the landscape's anisotropy the way seeding
    Powell with the covariance eigenvectors un-rotates its coordinate descent.

    **Shape** (relative anisotropy) comes from the covariance; **absolute size**
    is decoupled into ``base_size`` with a ``min_step`` floor.  This matters:
    the raw CMA-ES extent ``sigma * D`` *collapses* as CMA-ES converges, and a
    ``sigma * D``-sized simplex would satisfy Nelder-Mead's ``xatol`` / ``fatol``
    test on the first iteration and terminate having done nothing.  Pass
    ``absolute=True`` to use ``sigma * D`` anyway (only to study that collapse).

    Construction::

        vertex[0]   = clip(x0)
        vertex[k+1] = x0 + geometry.axis_steps(...)[:, k]

    reflected off **both** bounds (eigenvector edges are signed, unlike
    :class:`SimplexPopulationInitializer`'s always-positive axis steps) then
    clipped, with a per-edge fallback so tight bounds cannot collapse a vertex
    onto ``x0`` (which would give a degenerate zero-volume simplex).

    Deterministic: the ``rng`` argument is accepted for interface compatibility
    but never consumed.
    """

    def __init__(
        self,
        geometry: "InitialGeometry",
        base_size: float | None = None,
        base_fraction: float = 0.1,
        ratio_floor: float = 1e-3,
        min_step: float = 1e-6,
        normalize: bool = True,
        absolute: bool = False,
    ) -> None:
        self.geometry = geometry
        """The learned geometry whose principal axes shape the simplex."""
        self.base_size = base_size
        """Absolute length of the longest simplex edge. ``None`` = derive from
        the bounds at generation time (``base_fraction * mean(ub - lb)``)."""
        self.base_fraction = base_fraction
        """Fraction of the mean bound range used for ``base_size`` when it is
        not given explicitly."""
        self.ratio_floor = ratio_floor
        """Smallest allowed anisotropy ratio ``D_k / max(D)`` — caps how thin the
        steepest axis's edge may get relative to the widest."""
        self.min_step = min_step
        """Hard floor on every edge length. Set it above the optimizer's
        convergence tolerance (Nelder-Mead passes ``100 * xatol``) so the
        simplex never starts at the convergence tolerance."""
        self.normalize = normalize
        """Use relative anisotropy (shape only) with size decoupled into
        ``base_size``. This is the collapse-robust default."""
        self.absolute = absolute
        """Use the raw CMA-ES extent ``sigma * D`` for edge lengths (collapses as
        CMA-ES converges; for studies of that collapse only)."""

    def generate_population(
        self,
        rng: np.random.Generator,
        x0: NDArray[np.float64],
        pop_size: int,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del rng  # deterministic construction — interface compatibility only
        dim = len(x0)
        if pop_size != dim + 1:
            raise ValueError(
                f"A simplex over {dim} dimensions has exactly {dim + 1} "
                f"vertices; got pop_size={pop_size}."
            )

        x0 = np.clip(np.asarray(x0, dtype=float), lower_bounds, upper_bounds)

        base_size = self.base_size
        if base_size is None:
            span = upper_bounds - lower_bounds
            finite = np.isfinite(span)
            if np.any(finite):
                base_size = self.base_fraction * float(np.mean(span[finite]))
            else:
                base_size = self.base_fraction * max(1.0, float(np.linalg.norm(x0)))

        # (dim, dim) matrix whose column k is the edge vector for vertex k+1.
        steps = self.geometry.axis_steps(
            base_size=base_size,
            normalize=self.normalize,
            ratio_floor=self.ratio_floor,
            min_step=self.min_step,
            absolute=self.absolute,
        )

        sim = np.empty((dim + 1, dim), dtype=float)
        sim[0] = x0
        for k in range(dim):
            vertex = x0 + steps[:, k]
            # Reflect off both bounds (signed edges can overshoot either way),
            # then clip — plain clipping could collapse the edge onto a face.
            with np.errstate(invalid="ignore"):
                vertex = np.where(vertex > upper_bounds, 2 * upper_bounds - vertex, vertex)
                vertex = np.where(vertex < lower_bounds, 2 * lower_bounds - vertex, vertex)
            vertex = np.clip(vertex, lower_bounds, upper_bounds)
            # Degeneracy fallback: a vertex clipped back onto x0 collapses the
            # simplex. Nudge along coordinate k by min_step (into the interior).
            if float(np.linalg.norm(vertex - x0)) < 0.5 * self.min_step:
                vertex = x0.copy()
                step = self.min_step
                if vertex[k] + step > upper_bounds[k]:
                    step = -self.min_step
                vertex[k] = np.clip(x0[k] + step, lower_bounds[k], upper_bounds[k])
            sim[k + 1] = vertex
        return sim


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
    SIMPLEX = "simplex"
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
            case PopulationInitializerType.SIMPLEX:
                return SimplexPopulationInitializer()
            case PopulationInitializerType.IDENTITY:
                return IdentityPopulationInitializer()
