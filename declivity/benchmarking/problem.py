"""A test problem: function, dimensions, bounds, and a deterministic
starting-point generator keyed by seed.

Two algorithms run with the same seed start from the same x0.

The ``initial_point_generator`` field is pluggable:
- ``None`` (default) → ``UniformInitialPointGenerator``.
- an ``NDArray`` → wrapped in ``FixedInitialPointGenerator``, so the same
  fixed x0 is returned for every seed.
- any ``InitialPointGenerator`` subclass, used directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from declivity.cec import CECEdition, CECProblem
from declivity.utils.benchmark_functions import BenchmarkFunction
from declivity.utils.constraint_handlers import (
    BoxConstraintHandler,
    BoxStrategy,
    ConstraintHandler,
)
from declivity.utils.initial_point_generator import (
    FixedInitialPointGenerator,
    InitialPointGenerator,
    UniformInitialPointGenerator,
)


@dataclass
class Problem:
    """Test problem specification."""

    name: str
    """Short name used in plots and tables."""

    function: Callable[[NDArray[np.float64]], float]
    """Objective; usually a BenchmarkFunction subclass."""

    dimensions: int

    lower_bound: float
    upper_bound: float

    gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
    """Analytical gradient. If None, L-BFGS-B uses finite differences."""

    initial_point_generator: InitialPointGenerator | NDArray[np.float64] | None = None
    """Strategy for generating the starting point.

    - ``None`` (default): uniform sampling within ``[lower_bound, upper_bound]``.
    - ``NDArray``: always start from this fixed point regardless of seed.
    - ``InitialPointGenerator`` subclass: delegate to its ``generate_point``.
    """

    constraint_handler: ConstraintHandler | None = None
    """The problem's feasible region.

    ``None`` (default) means the box ``[lower_bound, upper_bound]`` with CLAMP
    repair.  Set it for a region the two scalar bounds cannot express: a
    per-dimension box, ``BoxStrategy.BOUNCE_BACK`` repair, or a custom
    :class:`~declivity.utils.constraint_handlers.ConstraintHandler` subclass.

    A :class:`~declivity.benchmarking.algorithm_run.BenchmarkAlgorithm` can
    override it when the handler itself is under study; see its
    ``constraint_handler`` field.

    Runners also forward ``lower_bound`` / ``upper_bound`` as the requested
    search box and :class:`BaseOptimizer` intersects the two, so a handler
    here can only tighten the region.
    """

    def __post_init__(self) -> None:
        # Normalise the initial_point_generator field so it is always an
        # InitialPointGenerator instance after construction.
        if self.initial_point_generator is None:
            self.initial_point_generator = UniformInitialPointGenerator()
        elif isinstance(self.initial_point_generator, np.ndarray):
            self.initial_point_generator = FixedInitialPointGenerator(
                self.initial_point_generator
            )

    @classmethod
    def from_benchmark(
        cls,
        name: str,
        function: BenchmarkFunction,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        initial_point_generator: InitialPointGenerator
        | NDArray[np.float64]
        | None = None,
        constraint_handler: ConstraintHandler | None = None,
    ) -> Problem:
        """Build a Problem from a BenchmarkFunction.

        Picks up the function's bounds and ``gradient`` attribute (if any)
        unless explicit values are passed. ``initial_point_generator`` and
        ``constraint_handler`` are forwarded verbatim (``None`` → uniform
        sampling within the bounds, and the default CLAMP box handler).
        """
        lb_array, ub_array = function.bounds
        # Only advertise an analytic gradient when the concrete function
        # overrides BenchmarkFunction.gradient; the base method raises
        # NotImplementedError.
        overrides_gradient = (
            getattr(type(function), "gradient", None) is not BenchmarkFunction.gradient
        )
        return cls(
            name=name,
            function=function,
            dimensions=function.dimensions,
            lower_bound=float(lb_array[0]) if lower_bound is None else lower_bound,
            upper_bound=float(ub_array[0]) if upper_bound is None else upper_bound,
            gradient=function.gradient if overrides_gradient else None,
            initial_point_generator=initial_point_generator,
            constraint_handler=constraint_handler,
        )

    @classmethod
    def from_cec(
        cls,
        edition: CECEdition,
        function_number: int,
        dimensions: int,
        *,
        initial_point_generator: InitialPointGenerator
        | NDArray[np.float64]
        | None = None,
        constraint_handler: ConstraintHandler | None = None,
    ) -> Problem:
        """Build a Problem from a cecxx-backed CEC benchmark function.

        See CECProblem for the (edition, function_number, dimensions)
        validity rules -- not every CEC edition supports the same
        dimensionalities.
        """
        function = CECProblem(edition, function_number, dimensions)
        return cls.from_benchmark(
            name=function.name,
            function=function,
            initial_point_generator=initial_point_generator,
            constraint_handler=constraint_handler,
        )

    def resolved_constraint_handler(self) -> ConstraintHandler:
        """The problem's feasible region as a concrete handler.

        Returns :attr:`constraint_handler` when one was given, otherwise
        ``BoxConstraintHandler(CLAMP, ...)`` over
        ``[lower_bound, upper_bound]``, so the starting-point generator and
        the optimizer use the same region.
        """
        if self.constraint_handler is not None:
            return self.constraint_handler
        return BoxConstraintHandler(
            BoxStrategy.CLAMP,
            np.full(self.dimensions, self.lower_bound, dtype=float),
            np.full(self.dimensions, self.upper_bound, dtype=float),
        )

    def starting_point(self, seed: int) -> NDArray[np.float64]:
        """Deterministic starting point inside the feasible region.

        Same seed => same x0, regardless of which algorithm consumes it.
        Delegates to ``self.initial_point_generator.generate_point``, handing
        it the resolved constraint handler so a non-box region is respected.
        """
        rng = np.random.default_rng(seed)
        assert isinstance(self.initial_point_generator, InitialPointGenerator)
        return self.initial_point_generator.generate_point(
            rng, self.dimensions, self.resolved_constraint_handler()
        )


class ProblemFamily:
    """One statistical problem, a different deterministic instance per seed.

    Wraps a ``seed -> Problem`` factory so studies that randomise the problem
    *instance* per seed — a fresh rotation or shift of the same objective —
    still flow through :class:`~declivity.benchmarking.Benchmark`: the runner
    resolves ``instance(seed)`` before each job, and every instance shares the
    family ``name`` so traces aggregate as one problem.

    All instances must share dimensions and bounds; metadata accessors
    (``dimensions``, ``function``, ...) read the template instance.
    """

    def __init__(
        self,
        name: str,
        factory: Callable[[int], Problem],
        template_seed: int = 0,
    ) -> None:
        self.name = name
        self._factory = factory
        self._template_seed = template_seed
        self._instances: dict[int, Problem] = {}

    def instance(self, seed: int) -> Problem:
        """The (cached) concrete :class:`Problem` for ``seed``."""
        if seed not in self._instances:
            problem = self._factory(seed)
            problem.name = self.name
            self._instances[seed] = problem
        return self._instances[seed]

    @property
    def template(self) -> Problem:
        """A representative instance, for metadata and plot annotations."""
        return self.instance(self._template_seed)

    @property
    def dimensions(self) -> int:
        return self.template.dimensions

    @property
    def lower_bound(self) -> float:
        return self.template.lower_bound

    @property
    def upper_bound(self) -> float:
        return self.template.upper_bound

    @property
    def function(self) -> Callable[[NDArray[np.float64]], float]:
        return self.template.function

    def resolved_constraint_handler(self) -> ConstraintHandler:
        return self.template.resolved_constraint_handler()

    def starting_point(self, seed: int) -> NDArray[np.float64]:
        return self.instance(seed).starting_point(seed)
