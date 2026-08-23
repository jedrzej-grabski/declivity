"""Algorithm specifications for benchmarking.

The framework only needs three things from anything you put inside a
:class:`~declivity.benchmarking.Benchmark`:

- ``name``  — appears in legends, summary tables, and persisted traces
- ``color`` — hex string the plotter uses for this algorithm's line
- ``run(problem, x0, seed)`` — returns a :class:`RunTrace`

Which base class to pick:

==============================  ===========================================
Your runner is...               Inherit from
==============================  ===========================================
one optimizer from the          :class:`SingleAlgorithm` (concrete).
factory, no special wrapping
warmup -> refinement, two       :class:`HandoffAlgorithm`. Implement
phases that share state via     ``run_phases()``; the base class stitches
local variables                 traces, clamps fitness, and fills in
                                handoff metadata.
anything else (3+ phases,       :class:`BenchmarkAlgorithm`. Implement
restarts, wrappers, custom      ``run()``; use ``trace_from_result()`` to
schedules)                      package an :class:`OptimizationResult`
                                into a :class:`RunTrace`.
already a class, can't          conform to the :class:`AlgorithmRun`
inherit                         :class:`Protocol`: expose ``name``,
                                ``color``, and ``run()``.
==============================  ===========================================
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.bfgs.bfgs_optimizer import BFGSOptimizer
from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.cmaes_optimizer import CMAESOptimizer, CMAESState
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.algorithms.lbfgsb.lbfgsb_optimizer import LBFGSBOptimizer
from declivity.benchmarking.problem import Problem
from declivity.benchmarking.run_trace import RunTrace, capture_scalar_series
from declivity.core.algorithm_factory import AlgorithmFactory
from declivity.core.base_optimizer import OptimizationResult
from declivity.core.config_base import BaseConfig
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.gradient_strategies import GradientStrategy
from declivity.utils.initial_geometry import (
    HandoffTransform,
    HessianScaling,
    InitialGeometry,
    covariance_to_hessian_matrix,
    scaling_factor,
)
from declivity.utils.line_search import LineSearchStrategy
from declivity.utils.population_initializers import PopulationInitializer
from declivity.utils.repair_strategies import RepairStrategy
from declivity.utils.stopping_conditions import (
    DEFAULT_EVALUATIONS_PER_DIMENSION,
    MaxEvaluations,
    StoppingCondition,
)


def initial_hessian_from_cmaes(
    transform: HandoffTransform | str,
    eigenvectors: NDArray[np.float64],
    eigenvalues_sqrt: NDArray[np.float64],
    scaling: HessianScaling | str = HessianScaling.NONE,
    sigma: float = 1.0,
    prev_norm: float | None = None,
) -> NDArray[np.float64] | None:
    """Turn a CMA-ES eigendecomposition ``(B, D)`` into an L-BFGS-B ``B_0``.

    ``C = B @ diag(D**2) @ B.T``.  The L-BFGS-B model needs the Hessian ``B``
    and the CMA-ES covariance ``C`` is proportional to ``B^{-1}``, so the
    useful transform inverts it.  See :class:`HandoffTransform`;
    ``IDENTITY`` returns ``None`` (the L-BFGS-B default ``B_0 = I``).

    ``scaling`` applies the :class:`HessianScaling` magnitude factor to the
    returned dense matrix (``sigma`` is only read by ``HessianScaling.SIGMA``,
    ``prev_norm`` only by ``HessianScaling.ADAPTIVE``).
    This seam feeds ``LBFGSBConfig.initial_hessian`` as a raw matrix rather
    than an :class:`InitialGeometry`, so the factor is baked in here; the
    numbers match :meth:`InitialGeometry.from_covariance` with the same
    ``transform`` / ``scaling`` (see :class:`CMAESLocalHandoff`).

    Delegates to
    :func:`~declivity.utils.initial_geometry.covariance_to_hessian_matrix`.
    """
    matrix = covariance_to_hessian_matrix(transform, eigenvectors, eigenvalues_sqrt)
    if matrix is None:
        return None
    factor = scaling_factor(scaling, matrix, matrix.shape[0], sigma, prev_norm=prev_norm)
    return matrix * factor


@runtime_checkable
class AlgorithmRun(Protocol):
    """Duck-typed interface for anything :class:`~declivity.benchmarking.Benchmark` consumes.

    Prefer subclassing :class:`BenchmarkAlgorithm`, which has the same
    three-attribute contract plus the
    :py:meth:`BenchmarkAlgorithm.trace_from_result` helper.  This Protocol is
    for classes that cannot inherit and only conform structurally.
    """

    name: str
    color: str

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace: ...


class BenchmarkAlgorithm(ABC):
    """Base class for anything you can plug into a :class:`Benchmark`.

    Subclasses provide:

    - ``name`` and ``color`` (typically as :py:func:`dataclasses.dataclass` fields).
    - :py:meth:`run`, which executes the algorithm on one
      ``(problem, x0, seed)`` triple and returns a :class:`RunTrace`.

    The :py:meth:`trace_from_result` helper packages a single
    :class:`OptimizationResult` into a :class:`RunTrace`.  Multi-phase
    runners with their own stitching rules assemble the trace by hand, or
    use :class:`HandoffAlgorithm` for the two-phase case.

    Minimal subclass::

        @dataclass
        class MyAlgorithm(BenchmarkAlgorithm):
            name: str
            color: str
            # config fields

            def run(self, problem, x0, seed) -> RunTrace:
                result = ...  # run an optimizer
                return self.trace_from_result(problem, seed, result)
    """

    name: str
    color: str

    constraint_handler: ConstraintHandler | None = None
    """Override for the problem's feasible region.

    ``None`` (default) uses :attr:`Problem.constraint_handler`.  Set this
    when the handler itself is under study, e.g. comparing
    ``BoxStrategy.CLAMP`` against ``BOUNCE_BACK`` on one problem definition.

    Read it through :meth:`resolve_constraint_handler` so the problem-level
    default is honoured.
    """

    retain_series: tuple[str, ...] | None = None
    """Which extra scalar-per-step diagnostics :meth:`trace_from_result`
    keeps on the trace, for cross-seed bands.  ``None`` (default) retains
    every cheap scalar field the LogData logged (``sigma``,
    ``condition_number``, ``mean_fitness``, ...); a tuple restricts capture
    to those fields; an empty tuple keeps only ``best_fitness``.  Heavy
    vector/matrix fields are never retained."""

    @abstractmethod
    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        """Run the algorithm on one (problem, x0, seed) triple."""
        ...

    def resolve_constraint_handler(self, problem: Problem) -> ConstraintHandler:
        """The handler this run should use: the runner's override, else the problem's.

        Custom :class:`BenchmarkAlgorithm` subclasses should call this and
        forward the result as ``constraint_handler=``; a runner that skips it
        pins the run to the default box.
        """
        if self.constraint_handler is not None:
            return self.constraint_handler
        return problem.resolved_constraint_handler()

    def trace_from_result(
        self,
        problem: Problem,
        seed: int,
        result: OptimizationResult,
    ) -> RunTrace:
        """Package a single :class:`OptimizationResult` into a :class:`RunTrace`.

        For the case where one optimizer call produces the whole convergence
        trace.  ``result.diagnostic`` must carry ``evaluations`` and
        ``best_fitness``.

        The trace is a trimmed ``LogData``: ``evaluations`` /
        ``best_fitness`` become first-class lists, and every other cheap
        scalar-per-step field (subject to :attr:`retain_series`) is captured
        into ``trace.series``.  Heavy per-iteration fields (population,
        eigenvalues, best_solution) are dropped.

        Multi-phase runners with custom stitching build the
        :class:`RunTrace` by hand, or use :class:`HandoffAlgorithm`.
        """
        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=list(result.diagnostic.evaluations),
            best_fitness=list(result.diagnostic.best_fitness),
            final_evaluations=result.evaluations,
            final_fitness=float(result.best_fitness),
            series=capture_scalar_series(result.diagnostic, retain=self.retain_series),
        )


@dataclass
class SingleAlgorithm(BenchmarkAlgorithm):
    """A single optimizer registered in the :class:`AlgorithmFactory`.

    Instantiate directly with the optimizer choice and a config factory.

    For evolutionary algorithms (DES, CMA-ES, MF-CMA-ES) a
    :class:`RepairStrategy` and / or :class:`PopulationInitializer` can be
    pinned to override the per-algorithm defaults.  These are forwarded to
    the factory only when set, so single-point algorithms are unaffected.
    """

    name: str
    color: str
    algorithm: AlgorithmChoice
    config_factory: Callable[[int], BaseConfig]
    """Builds a fresh config given the problem's dimensions."""

    extra_diagnostics: tuple[str, ...] = ()
    """Names of additional diag_* flags to enable on the config."""

    repair_strategy: RepairStrategy | None = None
    """Population-level repair policy; applies to evolutionary algorithms
    only.  ``None`` keeps the optimizer's default (``LamarckianRepair``)."""

    population_initializer: PopulationInitializer | None = None
    """How the iteration-0 population is seeded; applies to evolutionary
    algorithms only.  ``None`` keeps the optimizer's default
    (``NormalPopulationInitializer`` for DES,
    ``MeanSigmaPopulationInitializer(sigma=config.sigma)`` for CMA-ES and
    MF-CMA-ES)."""

    line_search: LineSearchStrategy | None = None
    """Line-search strategy for the gradient-based algorithms (L-BFGS-B,
    BFGS); ignored for the others. ``None`` keeps the optimizer's default
    (``MoreThuenteLineSearch``)."""

    gradient_strategy: GradientStrategy | None = None
    """Gradient-approximation strategy for the gradient-based algorithms
    (L-BFGS-B, BFGS); ignored for the others. ``None`` keeps the
    optimizer's default (``CentralFD``)."""

    stopping_condition: StoppingCondition | None = None
    """When to stop. ``None`` keeps the optimizer default
    (``MaxEvaluations(10_000 * dimensions)``). Pass e.g.
    ``MaxEvaluations(5000)``, ``MaxTime(30.0)``, or a composite
    (``MaxEvaluations(5000) | TargetFitness(1e-8)``) to override."""

    constraint_handler: ConstraintHandler | None = None
    """Feasible-region override; ``None`` uses the problem's.  See
    :attr:`BenchmarkAlgorithm.constraint_handler`."""

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        config = self.config_factory(problem.dimensions)
        for flag in self.extra_diagnostics:
            if hasattr(config, flag):
                setattr(config, flag, True)

        kwargs: dict = {}
        if self.stopping_condition is not None:
            kwargs["stopping_condition"] = self.stopping_condition
        # Both gradient-based optimizers take the same three seams, wired
        # uniformly so the comparison is fair.
        if self.algorithm in (AlgorithmChoice.LBFGSB, AlgorithmChoice.BFGS):
            if problem.gradient is not None:
                kwargs["gradient_fn"] = problem.gradient
            if self.line_search is not None:
                kwargs["line_search"] = self.line_search
            if self.gradient_strategy is not None:
                kwargs["gradient_strategy"] = self.gradient_strategy
        if self.repair_strategy is not None:
            kwargs["repair_strategy"] = self.repair_strategy
        if self.population_initializer is not None:
            kwargs["population_initializer"] = self.population_initializer

        result = AlgorithmFactory.create_optimizer(
            self.algorithm,
            problem.function,
            x0,
            config,
            constraint_handler=self.resolve_constraint_handler(problem),
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
            **kwargs,
        ).optimize()

        return self.trace_from_result(problem, seed, result)


class HandoffAlgorithm(BenchmarkAlgorithm):
    """Two-phase (warm-up -> refinement) handoff base class.

    Subclasses provide ``name`` and ``color`` and override
    :py:meth:`run_phases` to return the two phase results.  The base class
    fills in a :class:`RunTrace` with:

    - eval counts in the refinement segment offset by the warm-up's total
      evaluations, so the convergence trace is continuous;
    - the refinement segment's fitness clamped to never exceed the warm-up's
      best, since the refinement logger starts from ``f(x0_refinement)``;
    - ``handoff_eval`` and ``handoff_iter`` from the warm-up totals, which
      the plotter uses to draw handoff markers.

    Minimal usage::

        @dataclass
        class MyHandoff(HandoffAlgorithm):
            name: str
            color: str
            # config fields

            def run_phases(self, problem, x0, seed):
                warmup_result   = ...  # run phase 1
                refinement_result = ...  # run phase 2, using warmup state
                return warmup_result, refinement_result

    Both phases live in the same method, so warm-up state reaches
    refinement through ordinary local variables.

    For 3+ phases, restarts, or conditional branches, subclass
    :class:`BenchmarkAlgorithm` directly.
    """

    @abstractmethod
    def run_phases(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> tuple[OptimizationResult, OptimizationResult]:
        """Run the two phases. Return ``(warmup_result, refinement_result)``."""
        ...

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        warmup, refinement = self.run_phases(problem, x0, seed)
        return self._stitch_traces(problem, seed, warmup, refinement)

    def _stitch_traces(
        self,
        problem: Problem,
        seed: int,
        warmup: OptimizationResult,
        refinement: OptimizationResult,
    ) -> RunTrace:
        """Concatenate the two phases into a single continuous trace."""
        warmup_evals = warmup.evaluations
        warmup_iters = len(warmup.diagnostic.iteration)
        warmup_best = float(warmup.best_fitness)

        warmup_eval_list = list(warmup.diagnostic.evaluations)
        warmup_fitness_list = list(warmup.diagnostic.best_fitness)
        # Offset refinement eval counts so the trace is continuous.
        refinement_eval_list = [
            evaluation + warmup_evals
            for evaluation in refinement.diagnostic.evaluations
        ]
        # Clamp refinement fitness so it never reports worse than the
        # warm-up's best: its first point is f(x0_refinement), which can be
        # slightly worse than the warm-up's running minimum.
        refinement_fitness_list = [
            min(value, warmup_best) for value in refinement.diagnostic.best_fitness
        ]

        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=warmup_eval_list + refinement_eval_list,
            best_fitness=warmup_fitness_list + refinement_fitness_list,
            final_evaluations=warmup_evals + refinement.evaluations,
            final_fitness=min(warmup_best, float(refinement.best_fitness)),
            handoff_eval=warmup_evals,
            handoff_iter=warmup_iters,
        )


@dataclass
class CMAESLBFGSBHandoff(HandoffAlgorithm):
    """CMA-ES warm-up followed by L-BFGS-B with a covariance-derived ``B_0``.

    Steps:

    1. Run CMA-ES until :attr:`cmaes_stopping_condition` fires (default
       ``MaxEvaluations(10_000 * dimensions)``).  With the same seed and
       ``x0``, a standalone CMA-ES run produces an identical prefix.
    2. Read the cached eigendecomposition ``(B, D)`` from CMA-ES.
    3. Compose the initial Hessian for L-BFGS-B according to
       :attr:`transform` (see :class:`HandoffTransform`).
    4. Run L-BFGS-B from the CMA-ES mean with that ``B_0`` until
       :attr:`lbfgsb_stopping_condition` fires.

    Trace stitching, fitness clamping, and handoff metadata come from
    :class:`HandoffAlgorithm`.
    """

    name: str
    color: str
    cmaes_config_factory: Callable[[int], CMAESConfig]
    lbfgsb_config_factory: Callable[[int], LBFGSBConfig]
    transform: HandoffTransform | str = HandoffTransform.INVERSE
    scaling: HessianScaling | str = HessianScaling.NONE
    """Magnitude factor applied on top of ``transform`` (see
    :class:`HessianScaling`).  ``NONE`` keeps the raw ``C^{-1}``."""

    cmaes_stopping_condition: StoppingCondition | None = None
    """Warm-up (CMA-ES) stopping condition. ``None`` keeps the optimizer
    default (``MaxEvaluations(10_000 * dimensions)``); pass e.g.
    ``MaxEvaluations(warmup_budget)`` to bound the warm-up phase."""

    lbfgsb_stopping_condition: StoppingCondition | None = None
    """Refinement (L-BFGS-B) stopping condition. ``None`` keeps the
    optimizer default; pass ``MaxEvaluations(refinement_budget)`` to bound
    the refinement phase."""

    cmaes_extra_diagnostics: tuple[str, ...] = ("diag_eigen",)
    lbfgsb_extra_diagnostics: tuple[str, ...] = ()

    lbfgsb_line_search: LineSearchStrategy | None = None
    """Line-search strategy for the L-BFGS-B refinement phase. ``None``
    keeps the optimizer default (``MoreThuenteLineSearch``)."""

    lbfgsb_gradient_strategy: GradientStrategy | None = None
    """Gradient-approximation strategy for the L-BFGS-B refinement
    phase. ``None`` keeps the optimizer default (``CentralFD``)."""

    constraint_handler: ConstraintHandler | None = None
    """Feasible-region override, applied to both phases.  ``None`` uses the
    problem's."""

    def _initial_hessian_from_cmaes(
        self,
        eigenvectors: NDArray[np.float64],
        eigenvalues_sqrt: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.float64] | None:
        return initial_hessian_from_cmaes(
            self.transform,
            eigenvectors,
            eigenvalues_sqrt,
            scaling=self.scaling,
            sigma=sigma,
        )

    def run_phases(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> tuple[OptimizationResult, OptimizationResult]:
        # Phase 1: CMA-ES warm-up.
        cmaes_config = self.cmaes_config_factory(problem.dimensions)
        for flag in self.cmaes_extra_diagnostics:
            if hasattr(cmaes_config, flag):
                setattr(cmaes_config, flag, True)

        handler = self.resolve_constraint_handler(problem)

        _raw_cmaes_optimizer = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.CMAES,
            problem.function,
            x0,
            cmaes_config,
            constraint_handler=handler,
            stopping_condition=self.cmaes_stopping_condition,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
        )
        assert isinstance(_raw_cmaes_optimizer, CMAESOptimizer)
        cmaes_optimizer = _raw_cmaes_optimizer
        cmaes_result = cmaes_optimizer.optimize()

        # Pull CMA-ES internal state for the handoff: the eigendecomposition
        # of the covariance and the current mean become L-BFGS-B's B_0
        # and starting point respectively.
        eigenvectors, eigenvalues_sqrt = cmaes_optimizer.get_eigendecomposition()  # type: ignore[union-attr]
        initial_hessian = self._initial_hessian_from_cmaes(
            eigenvectors,
            eigenvalues_sqrt,
            cmaes_optimizer.sigma,  # type: ignore[union-attr]
        )

        # Phase 2: L-BFGS-B from the CMA-ES mean with the derived B_0.
        lbfgsb_config = self.lbfgsb_config_factory(problem.dimensions)
        lbfgsb_config.initial_hessian = initial_hessian
        for flag in self.lbfgsb_extra_diagnostics:
            if hasattr(lbfgsb_config, flag):
                setattr(lbfgsb_config, flag, True)

        lbfgsb_kwargs: dict = {}
        if problem.gradient is not None:
            lbfgsb_kwargs["gradient_fn"] = problem.gradient
        if self.lbfgsb_line_search is not None:
            lbfgsb_kwargs["line_search"] = self.lbfgsb_line_search
        if self.lbfgsb_gradient_strategy is not None:
            lbfgsb_kwargs["gradient_strategy"] = self.lbfgsb_gradient_strategy

        lbfgsb_result = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.LBFGSB,
            problem.function,
            cmaes_optimizer.mean,  # type: ignore[union-attr]
            lbfgsb_config,
            constraint_handler=handler,
            stopping_condition=self.lbfgsb_stopping_condition,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            **lbfgsb_kwargs,
        ).optimize()

        return cmaes_result, lbfgsb_result


@dataclass
class CMAESLocalHandoff(HandoffAlgorithm):
    """CMA-ES warm-up followed by a local optimizer seeded from the covariance.

    The generalization of :class:`CMAESLBFGSBHandoff` to any of the three
    single-point local optimizers. After a CMA-ES warm-up it builds **one**
    :class:`~declivity.algorithms.lbfgsb.initial_hessian.InitialGeometry` from
    the learned covariance and hands it to the chosen ``local_algorithm`` through
    the uniform ``initial_geometry=`` seam:

    - ``LBFGSB``     — the geometry is the initial Hessian ``B_0`` (``= C^{-1}``).
    - ``BFGS``       — the same curvature, inverted once more: BFGS tracks the
      *inverse* Hessian, so it seeds ``H_0 = B_0^{-1}``.
    - ``POWELL``     — its eigenvectors become the initial search-direction set.
    - ``NELDERMEAD`` — its principal axes shape the initial simplex.

    ``transform=INVERSE`` (curvature ``C^{-1}``) is the correct default for all
    four: the quasi-Newton pair needs the inverse, while Powell / Nelder-Mead
    read only the eigenvectors / anisotropy ratios (invariant to the
    inversion).  ``IDENTITY``
    is the control — an isotropic geometry (identity ``B_0`` / coordinate
    directions / isotropic simplex) that still flows through the same seam, so it
    isolates "covariance information" from "shared warm-up ``x0``".

    Same seed and ``x0`` as a standalone CMA-ES run produce an identical CMA-ES
    prefix in the trace. Trace stitching, fitness clamping, and handoff metadata
    are inherited from :class:`HandoffAlgorithm`; this class only owns the
    covariance transform and the per-target seam wiring.
    """

    name: str
    color: str
    local_algorithm: AlgorithmChoice
    cmaes_config_factory: Callable[[int], CMAESConfig]
    local_config_factory: Callable[[int], BaseConfig]
    transform: HandoffTransform | str = HandoffTransform.INVERSE
    scaling: HessianScaling | str = HessianScaling.NONE
    """Magnitude factor applied on top of ``transform`` (see
    :class:`HessianScaling`); orthogonal to the shape choice."""

    cmaes_stopping_condition: StoppingCondition | None = None
    """Warm-up (CMA-ES) stopping condition. ``None`` keeps the optimizer default
    (``MaxEvaluations(10_000 * dimensions)``); pass e.g.
    ``MaxEvaluations(warmup_budget)`` to bound the warm-up phase."""

    local_stopping_condition: StoppingCondition | None = None
    """Refinement (local optimizer) stopping condition. ``None`` keeps the
    optimizer default; pass ``MaxEvaluations(refinement_budget)`` to bound it."""

    cmaes_extra_diagnostics: tuple[str, ...] = ("diag_eigen",)
    local_extra_diagnostics: tuple[str, ...] = ()

    simplex_base_size: float | None = None
    """Nelder-Mead only: absolute length of the longest simplex edge. ``None``
    derives it from the bounds. Ignored by the other targets."""

    local_line_search: LineSearchStrategy | None = None
    """Gradient-based targets (L-BFGS-B, BFGS) only: line-search strategy for
    the refinement phase. ``None`` keeps the optimizer default
    (``MoreThuenteLineSearch``)."""

    local_gradient_strategy: GradientStrategy | None = None
    """Gradient-based targets (L-BFGS-B, BFGS) only: gradient-approximation
    strategy for the refinement phase. ``None`` keeps the optimizer default
    (``CentralFD``)."""

    constraint_handler: ConstraintHandler | None = None
    """Feasible-region override, applied to both phases.  ``None`` uses the
    problem's."""

    def run_phases(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> tuple[OptimizationResult, OptimizationResult]:
        valid_targets = (
            AlgorithmChoice.LBFGSB,
            AlgorithmChoice.BFGS,
            AlgorithmChoice.POWELL,
            AlgorithmChoice.NELDERMEAD,
        )
        if self.local_algorithm not in valid_targets:
            names = ", ".join(str(a) for a in valid_targets)
            raise ValueError(
                f"CMAESLocalHandoff.local_algorithm must be one of {names}; "
                f"got {self.local_algorithm!r}."
            )

        # Phase 1: CMA-ES warm-up.
        cmaes_config = self.cmaes_config_factory(problem.dimensions)
        for flag in self.cmaes_extra_diagnostics:
            if hasattr(cmaes_config, flag):
                setattr(cmaes_config, flag, True)

        handler = self.resolve_constraint_handler(problem)

        _raw_cmaes_optimizer = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.CMAES,
            problem.function,
            x0,
            cmaes_config,
            constraint_handler=handler,
            stopping_condition=self.cmaes_stopping_condition,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
        )
        assert isinstance(_raw_cmaes_optimizer, CMAESOptimizer)
        cmaes_optimizer = _raw_cmaes_optimizer
        cmaes_result = cmaes_optimizer.optimize()

        # One geometry object from the learned covariance.  INVERSE gives the
        # curvature B_0 = C^{-1}; Powell and Nelder-Mead take only its
        # eigenvectors / anisotropy, L-BFGS-B the matrix itself.
        eigenvectors, eigenvalues_sqrt = cmaes_optimizer.get_eigendecomposition()
        geometry = InitialGeometry.from_covariance(
            eigenvectors,
            eigenvalues_sqrt,
            cmaes_optimizer.sigma,
            self.transform,
            scaling=self.scaling,
        )

        # Phase 2: local optimizer from the CMA-ES mean, seeded by the geometry.
        local_config = self.local_config_factory(problem.dimensions)
        for flag in self.local_extra_diagnostics:
            if hasattr(local_config, flag):
                setattr(local_config, flag, True)

        local_kwargs: dict = {"initial_geometry": geometry}
        if self.local_algorithm in (AlgorithmChoice.LBFGSB, AlgorithmChoice.BFGS):
            if problem.gradient is not None:
                local_kwargs["gradient_fn"] = problem.gradient
            if self.local_line_search is not None:
                local_kwargs["line_search"] = self.local_line_search
            if self.local_gradient_strategy is not None:
                local_kwargs["gradient_strategy"] = self.local_gradient_strategy
        elif self.local_algorithm == AlgorithmChoice.NELDERMEAD:
            if self.simplex_base_size is not None:
                local_kwargs["simplex_base_size"] = self.simplex_base_size

        local_result = AlgorithmFactory.create_optimizer(
            self.local_algorithm,
            problem.function,
            cmaes_optimizer.mean,
            local_config,
            constraint_handler=handler,
            stopping_condition=self.local_stopping_condition,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            **local_kwargs,
        ).optimize()

        return cmaes_result, local_result


@dataclass
class InterleaveResult:
    """Detailed record of one :class:`InterleavedCMAESLocal` run.

    :attr:`trace` is the standard :class:`RunTrace` that
    :class:`~declivity.benchmarking.Benchmark` consumes.  The remaining
    fields expose the run's internal structure for
    :func:`declivity.plotting.plot_interleaved_convergence`:

    - the CMA-ES backbone: best-so-far over CMA-ES generations only,
      ignoring the local-optimizer drops;
    - each local-optimizer burst as its own ``(evaluations, best)`` segment;
    - the cumulative evaluation counts at which bursts began.
    """

    trace: RunTrace
    overall_evaluations: list[int]
    overall_best: list[float]
    cmaes_evaluations: list[int]
    cmaes_best: list[float]
    burst_segments: list[tuple[list[int], list[float]]]
    burst_starts: list[int]
    cmaes_generations: int
    num_bursts: int


@dataclass
class InterleavedCMAESLocal(BenchmarkAlgorithm):
    """Alternating CMA-ES <-> local optimizer with a covariance-derived seed.

    Where :class:`CMAESLocalHandoff` cuts over once, this cycles between
    CMA-ES and a chosen single-point local optimizer (``local_algorithm``,
    L-BFGS-B or BFGS) for the whole budget:

    1. Advance CMA-ES for ``cmaes_interval`` generations, resumed from its
       own :class:`CMAESState` each cycle.  With a shared RNG this
       reproduces a standalone CMA-ES run bit-for-bit.
    2. Fire a local-optimizer side-probe from the current CMA-ES mean, with
       its curvature ``B_0`` derived from the CMA-ES covariance
       (``transform``, default ``C^{-1}``) and magnitude set by ``scaling``
       (see :class:`HessianScaling`). The probe runs until its own
       convergence test fires (L-BFGS-B: ``probe_factr`` relative-decrease;
       BFGS: ``probe_pgtol`` as its gradient tolerance, since BFGS has no
       relative-decrease test), capped by ``probe_max_evals``.
    3. Fold the probe's improvements into the tracked overall best and
       return to step 1 with CMA-ES untouched.  The probe never feeds back
       into the CMA-ES distribution.

    ``scaling=HessianScaling.ADAPTIVE`` carries the *previous* probe's
    effective magnitude into the next one instead of a fixed formula, so a
    single choice works across dimensions / conditioning without sweeping a
    scale by hand: after each probe, the magnitude actually reached in
    whichever space that optimizer natively tracks (``theta * ||B_0||`` for
    L-BFGS-B's compact representation; ``||H_final||`` for BFGS's dense
    inverse Hessian) becomes ``prev_norm`` for the next probe.

    The trace is a staircase: CMA-ES descends gently while each probe drops
    the overall best toward the local minimum of the region CMA-ES
    currently occupies.

    Implemented on :class:`BenchmarkAlgorithm` rather than
    :class:`HandoffAlgorithm`, which is strictly two-phase.
    """

    name: str
    color: str
    cmaes_config_factory: Callable[[int], CMAESConfig]
    local_config_factory: Callable[[int], BaseConfig]

    local_algorithm: AlgorithmChoice = AlgorithmChoice.LBFGSB
    """Which single-point local optimizer runs each probe: ``LBFGSB`` or
    ``BFGS``."""

    cmaes_interval: int = 20
    """CMA-ES generations to run between consecutive probes (the handoff
    interval ``N``)."""

    total_budget: int = 0
    """Shared evaluation budget for the whole run. ``0`` -> use the CMA-ES
    config factory's own budget."""

    transform: HandoffTransform | str = HandoffTransform.INVERSE
    """How each probe turns the CMA-ES covariance into its ``B_0``.
    Default ``INVERSE`` (``C^{-1}``) passes covariance information only,
    matching :class:`CMAESLocalHandoff`."""

    scaling: HessianScaling | str = HessianScaling.NONE
    """Magnitude factor applied on top of ``transform`` for each probe's
    ``B_0`` (see :class:`HessianScaling`).  ``NONE`` keeps the raw ``C^{-1}``;
    ``ADAPTIVE`` carries the previous probe's effective scale forward."""

    probe_factr: float = 1e7
    """L-BFGS-B relative-decrease stop for each probe: the burst hands back
    once ``(f_old - f_new)/max(|f_old|,|f_new|,1) <= probe_factr * eps``.
    Larger values end the burst sooner; smaller ones grind closer to the
    local minimum. Ignored when ``local_algorithm=BFGS`` (no such test)."""

    probe_pgtol: float = 1e-8
    """Projected-gradient stop for each L-BFGS-B probe, or the plain
    gradient-norm stop (``BFGSConfig.gtol``) for each BFGS probe."""

    probe_max_evals: int = 1000
    """Hard cap on evaluations per probe, so a single burst cannot consume
    the whole remaining budget."""

    cmaes_extra_diagnostics: tuple[str, ...] = ()
    local_extra_diagnostics: tuple[str, ...] = ()

    local_line_search: LineSearchStrategy | None = None
    """Line-search strategy for the probes.  ``None`` keeps the optimizer
    default (``MoreThuenteLineSearch``)."""

    local_gradient_strategy: GradientStrategy | None = None

    constraint_handler: ConstraintHandler | None = None
    """Feasible-region override, shared by the CMA-ES backbone and every
    probe. ``None`` uses the problem's."""

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        return self.run_with_detail(problem, x0, seed).trace

    def run_with_detail(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> InterleaveResult:
        """Run the interleaved scheme and return the full diagnostic record.

        :py:meth:`run` is the framework entry point and returns only
        ``.trace``; use this when you want the backbone / burst structure
        for the staircase plot.
        """
        valid_targets = (AlgorithmChoice.LBFGSB, AlgorithmChoice.BFGS)
        if self.local_algorithm not in valid_targets:
            names = ", ".join(str(a) for a in valid_targets)
            raise ValueError(
                f"InterleavedCMAESLocal.local_algorithm must be one of {names}; "
                f"got {self.local_algorithm!r}."
            )

        rng = np.random.default_rng(seed)
        dimensions = problem.dimensions
        handler = self.resolve_constraint_handler(problem)

        reference_config = self.cmaes_config_factory(dimensions)
        population_size = reference_config.population_size
        total_budget = (
            self.total_budget
            if self.total_budget > 0
            else DEFAULT_EVALUATIONS_PER_DIMENSION * dimensions
        )
        evaluations_per_generation = population_size + 1
        interval_evaluations = max(
            evaluations_per_generation,
            self.cmaes_interval * evaluations_per_generation,
        )

        state: CMAESState | None = None
        cumulative = 0
        overall_best = float("inf")
        cmaes_best = float("inf")

        overall_evaluations: list[int] = []
        overall_best_curve: list[float] = []
        cmaes_evaluations: list[int] = []
        cmaes_best_curve: list[float] = []
        burst_segments: list[tuple[list[int], list[float]]] = []
        burst_starts: list[int] = []

        total_generations = 0
        cmaes_converged = False
        previous_scale: float | None = None

        while cumulative < total_budget:
            # CMA-ES slice, resumed from its own state.
            slice_budget = min(interval_evaluations, total_budget - cumulative)
            if slice_budget < evaluations_per_generation:
                break

            cmaes_config = self.cmaes_config_factory(dimensions)
            for flag in self.cmaes_extra_diagnostics:
                if hasattr(cmaes_config, flag):
                    setattr(cmaes_config, flag, True)

            cmaes = CMAESOptimizer(
                problem.function,
                x0,
                cmaes_config,
                constraint_handler=handler,
                stopping_condition=MaxEvaluations(slice_budget),
                lower_bounds=problem.lower_bound,
                upper_bounds=problem.upper_bound,
                seed=rng,
                initial_state=state,
            )
            cmaes_result = cmaes.optimize()

            for evaluation, fitness in zip(
                cmaes_result.diagnostic.evaluations,
                cmaes_result.diagnostic.best_fitness,
            ):
                global_evaluation = cumulative + int(evaluation)
                cmaes_best = min(cmaes_best, float(fitness))
                overall_best = min(overall_best, float(fitness))
                cmaes_evaluations.append(global_evaluation)
                cmaes_best_curve.append(cmaes_best)
                overall_evaluations.append(global_evaluation)
                overall_best_curve.append(overall_best)

            cumulative += cmaes_result.evaluations
            total_generations += len(cmaes_result.diagnostic.iteration)
            state = cmaes.get_state()

            # A non-budget termination means CMA-ES converged or stalled; do
            # one last probe then stop, otherwise the loop keeps re-hitting
            # the same criterion.
            if not cmaes_result.message.startswith("Maximum function evaluations"):
                cmaes_converged = True

            if cumulative >= total_budget:
                break

            # Local-optimizer side-probe from the current CMA-ES mean.
            probe_budget = min(self.probe_max_evals, total_budget - cumulative)
            if probe_budget <= 0:
                break

            eigenvectors, eigenvalues_sqrt = cmaes.get_eigendecomposition()

            # L-BFGS-B consumes the curvature B_0 directly, so ``scaling`` /
            # ``prev_norm`` act on B_0's own space. BFGS instead tracks the
            # *inverse* Hessian H_0; scaling B_0 and inverting it afterwards
            # would target the wrong space (norm(inv(M)) != 1/norm(M) for a
            # non-scalar M), so for BFGS the same ``scaling_factor`` call is
            # applied directly to the H_0 shape, then inverted once to get
            # the B_0 :class:`InitialGeometry` stores canonically.
            b0_shape = covariance_to_hessian_matrix(
                self.transform, eigenvectors, eigenvalues_sqrt
            )
            if self.local_algorithm == AlgorithmChoice.LBFGSB:
                target_shape = (
                    b0_shape if b0_shape is not None else np.eye(dimensions)
                )
                factor = scaling_factor(
                    self.scaling,
                    target_shape,
                    dimensions,
                    cmaes.sigma,
                    prev_norm=previous_scale,
                )
                curvature = target_shape * factor
            else:
                h0_shape = (
                    np.linalg.inv(b0_shape) if b0_shape is not None
                    else np.eye(dimensions)
                )
                factor = scaling_factor(
                    self.scaling,
                    h0_shape,
                    dimensions,
                    cmaes.sigma,
                    prev_norm=previous_scale,
                )
                inverse_hessian = h0_shape * factor
                curvature = np.linalg.inv(inverse_hessian)
                curvature = 0.5 * (curvature + curvature.T)
            geometry = InitialGeometry.from_curvature(curvature, dimensions)

            local_config = self.local_config_factory(dimensions)
            if self.local_algorithm == AlgorithmChoice.LBFGSB:
                local_config.factr = self.probe_factr
                local_config.pgtol = self.probe_pgtol
            else:
                # BFGS has no relative-decrease ("factr") test; its gradient
                # tolerance is the closest analog to the probe's stop.
                local_config.gtol = self.probe_pgtol
            for flag in self.local_extra_diagnostics:
                if hasattr(local_config, flag):
                    setattr(local_config, flag, True)

            local_kwargs: dict = {"initial_geometry": geometry}
            if problem.gradient is not None:
                local_kwargs["gradient_fn"] = problem.gradient
            if self.local_line_search is not None:
                local_kwargs["line_search"] = self.local_line_search
            if self.local_gradient_strategy is not None:
                local_kwargs["gradient_strategy"] = self.local_gradient_strategy

            probe = AlgorithmFactory.create_optimizer(
                self.local_algorithm,
                problem.function,
                cmaes.mean,
                local_config,
                constraint_handler=handler,
                stopping_condition=MaxEvaluations(probe_budget),
                lower_bounds=problem.lower_bound,
                upper_bounds=problem.upper_bound,
                **local_kwargs,
            )
            probe_result = probe.optimize()

            # Effective magnitude this burst actually converged to, in
            # whichever space this optimizer natively tracks, for ADAPTIVE
            # scaling to carry into the next probe. L-BFGS-B's compact
            # representation scales B_0 by theta over the run; BFGS exposes
            # its dense H_k directly.
            if self.local_algorithm == AlgorithmChoice.LBFGSB:
                assert isinstance(probe, LBFGSBOptimizer)
                _, _, theta = probe.final_corrections()
                previous_scale = theta * float(np.linalg.norm(curvature))
            else:
                assert isinstance(probe, BFGSOptimizer)
                h_final = probe.final_inverse_hessian
                if h_final is not None:
                    previous_scale = float(np.linalg.norm(h_final))

            burst_starts.append(cumulative)
            # Start the burst segment at the current overall-best level so
            # the drop renders as a clean step.  The values are the running
            # best within the probe.
            burst_running = overall_best
            segment_evaluations = [cumulative]
            segment_best = [burst_running]
            for evaluation, fitness in zip(
                probe_result.diagnostic.evaluations,
                probe_result.diagnostic.best_fitness,
            ):
                global_evaluation = cumulative + int(evaluation)
                burst_running = min(burst_running, float(fitness))
                overall_best = min(overall_best, float(fitness))
                segment_evaluations.append(global_evaluation)
                segment_best.append(burst_running)
                overall_evaluations.append(global_evaluation)
                overall_best_curve.append(overall_best)
            burst_segments.append((segment_evaluations, segment_best))

            cumulative += probe_result.evaluations

            if cmaes_converged:
                break

        trace = RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=overall_evaluations,
            best_fitness=overall_best_curve,
            final_evaluations=cumulative,
            final_fitness=overall_best,
            handoff_eval=(burst_starts[0] if burst_starts else None),
            handoff_iter=(self.cmaes_interval if burst_starts else None),
        )
        return InterleaveResult(
            trace=trace,
            overall_evaluations=overall_evaluations,
            overall_best=overall_best_curve,
            cmaes_evaluations=cmaes_evaluations,
            cmaes_best=cmaes_best_curve,
            burst_segments=burst_segments,
            burst_starts=burst_starts,
            cmaes_generations=total_generations,
            num_bursts=len(burst_starts),
        )
