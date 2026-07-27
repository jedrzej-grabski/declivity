"""Algorithm specifications for benchmarking.

The framework only needs three things from anything you put inside a
:class:`~src.benchmarking.Benchmark`:

- ``name``  — appears in legends, summary tables, and persisted traces
- ``color`` — hex string the plotter uses for this algorithm's line
- ``run(problem, x0, seed)`` — returns a :class:`RunTrace`

How you provide them determines which base class to pick:

==============================  ===========================================
Your runner is...               Inherit from
==============================  ===========================================
one optimizer from the          :class:`SingleAlgorithm` (concrete, just
factory, no special wrapping    instantiate it).
warmup -> refinement, two       :class:`HandoffAlgorithm`. Implement
phases that share state via     ``run_phases()``; the base class stitches
local variables                 traces, clamps fitness, and fills in
                                handoff metadata for you.
anything else (3+ phases,       :class:`BenchmarkAlgorithm`. Implement
restarts, wrappers, custom      ``run()``; use ``trace_from_result()`` to
schedules)                      package an :class:`OptimizationResult`
                                into a :class:`RunTrace`.
already a class, can't          conform to the :class:`AlgorithmRun`
inherit                         :class:`Protocol` — just expose ``name``,
                                ``color``, and ``run()``. No inheritance
                                required.
==============================  ===========================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.cmaes_optimizer import CMAESOptimizer, CMAESState
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.algorithms.lbfgsb.lbfgsb_optimizer import LBFGSBOptimizer
from declivity.utils.line_search import LineSearchStrategy
from declivity.benchmarking.problem import Problem
from declivity.benchmarking.run_trace import RunTrace, capture_scalar_series
from declivity.core.algorithm_factory import AlgorithmFactory
from declivity.core.base_optimizer import OptimizationResult
from declivity.core.config_base import BaseConfig
from declivity.utils.gradient_strategies import GradientStrategy
from declivity.utils.population_initializers import PopulationInitializer
from declivity.utils.repair_strategies import RepairStrategy
from declivity.utils.stopping_conditions import (
    DEFAULT_EVALUATIONS_PER_DIMENSION,
    MaxEvaluations,
    StoppingCondition,
)


_EIGENVALUE_FLOOR = 1e-30


class HandoffTransform(StrEnum):
    """How to turn the CMA-ES covariance into an L-BFGS-B initial Hessian.

    Used by :class:`CMAESLBFGSBHandoff`.
    """

    INVERSE = "inverse"
    """Use ``C^{-1}`` directly. The L-BFGS-B model becomes a true quadratic
    approximation of the CMA-ES posterior around the warm-up mean."""

    SIGMA_INVERSE = "sigma_inverse"
    """Use ``(sigma^2 C)^{-1}`` — accounts for the CMA-ES global step-size
    scaling. Sometimes more conservative than the bare inverse."""

    IDENTITY = "identity"
    """Drop the covariance and use the L-BFGS-B default (B_0 = I). Mainly a
    control experiment: isolates the value of *passing covariance information*
    from the value of *sharing a starting point* with CMA-ES."""


def initial_hessian_from_cmaes(
    transform: HandoffTransform | str,
    eigenvectors: NDArray[np.float64],
    eigenvalues_sqrt: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64] | None:
    """Turn a CMA-ES eigendecomposition ``(B, D)`` into an L-BFGS-B ``B_0``.

    ``C = B @ diag(D**2) @ B.T``; the L-BFGS-B model needs ``B`` (the
    Hessian), and the CMA-ES covariance ``C`` is proportional to
    ``B^{-1}`` — so the useful transforms invert it. Shared by both the
    one-shot :class:`CMAESLBFGSBHandoff` and the
    :class:`InterleavedCMAESLBFGSB` scheme. See :class:`HandoffTransform`
    for the meaning of each option; ``IDENTITY`` returns ``None`` (the
    L-BFGS-B default ``B_0 = I``).
    """
    transform = str(transform)
    if transform == HandoffTransform.IDENTITY:
        return None

    eigenvalues = np.maximum(eigenvalues_sqrt**2, _EIGENVALUE_FLOOR)

    # Floored 1/eigenvalues can be huge (up to 1e30); the matmul values are
    # still valid but numpy raises spurious divide/overflow warnings on the
    # intermediates.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if transform == HandoffTransform.INVERSE:
            return (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
        if transform == HandoffTransform.SIGMA_INVERSE:
            inverse = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
            return inverse / (sigma * sigma)

    valid = ", ".join(repr(value.value) for value in HandoffTransform)
    raise ValueError(
        f"Unknown handoff transform: {transform!r}. Use one of {valid}."
    )


@runtime_checkable
class AlgorithmRun(Protocol):
    """Duck-typed interface for anything :class:`~src.benchmarking.Benchmark` consumes.

    For most use cases prefer subclassing :class:`BenchmarkAlgorithm` —
    same three-attribute contract, plus you get the
    :py:meth:`BenchmarkAlgorithm.trace_from_result` helper. The Protocol
    is here for the case where you can't inherit (e.g. you're adapting
    an existing class) and just want to conform structurally.
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
    :class:`OptimizationResult` into a :class:`RunTrace` for the common
    one-optimizer case. Multi-phase runners with their own stitching
    rules (handoffs, restarts) skip the helper and assemble the trace
    by hand, or use :class:`HandoffAlgorithm` for the standard two-phase
    pattern.

    Minimal subclass::

        @dataclass
        class MyAlgorithm(BenchmarkAlgorithm):
            name: str
            color: str
            # ... whatever config fields you need ...

            def run(self, problem, x0, seed) -> RunTrace:
                result = ...  # run an optimizer
                return self.trace_from_result(problem, seed, result)
    """

    name: str
    color: str

    retain_series: tuple[str, ...] | None = None
    """Which extra scalar-per-step diagnostics :meth:`trace_from_result`
    keeps on the trace (for cross-seed bands). ``None`` (default) = "auto":
    retain every cheap scalar field the LogData logged (``sigma``,
    ``condition_number``, ``mean_fitness``, ...). A tuple restricts capture
    to exactly those fields; an empty tuple keeps only ``best_fitness``
    (the old lean behavior). Heavy vector/matrix fields are never retained.
    This is the storage-side knob for the single-plotter design: whatever is
    retained here becomes available to the benchmark band plotter under the
    same panel that drew it on a single run."""

    @abstractmethod
    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        """Run the algorithm on one (problem, x0, seed) triple."""
        ...

    def trace_from_result(
        self,
        problem: Problem,
        seed: int,
        result: OptimizationResult,
    ) -> RunTrace:
        """Package a single :class:`OptimizationResult` into a :class:`RunTrace`.

        Handles the common case where one optimizer call produces the
        whole convergence trace. ``result.diagnostic`` is expected to
        carry ``evaluations`` and ``best_fitness`` (every built-in
        LogData does).

        The trace is a *trimmed* ``LogData``: ``evaluations`` /
        ``best_fitness`` become first-class lists, and every other cheap
        scalar-per-step field (subject to :attr:`retain_series`) is captured
        into ``trace.series`` so the same panel that plotted it on a single
        run can plot an aggregated band across a benchmark's seeds. Heavy
        per-iteration fields (population, eigenvalues, best_solution) are
        dropped — they don't persist at scale and can't be aggregated.

        For multi-phase runners with custom stitching rules, build the
        :class:`RunTrace` by hand instead — or use
        :class:`HandoffAlgorithm`, which handles the two-phase case.
        """
        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=list(result.diagnostic.evaluations),
            best_fitness=list(result.diagnostic.best_fitness),
            final_evaluations=result.evaluations,
            final_fitness=float(result.best_fitness),
            series=capture_scalar_series(
                result.diagnostic, retain=self.retain_series
            ),
        )


@dataclass
class SingleAlgorithm(BenchmarkAlgorithm):
    """A single optimizer registered in the :class:`AlgorithmFactory`.

    Concrete — instantiate directly with the optimizer choice and a
    config factory; no subclass needed unless you want to override
    :py:meth:`run`.

    For evolutionary algorithms (DES, CMA-ES, MF-CMA-ES) you can
    optionally pin a :class:`RepairStrategy` and / or
    :class:`PopulationInitializer` to override the per-algorithm
    defaults. These are forwarded to the factory only when set, so
    single-point algorithms like L-BFGS-B (which do not accept them) are
    unaffected.
    """

    name: str
    color: str
    algorithm: AlgorithmChoice
    config_factory: Callable[[int], BaseConfig]
    """Builds a fresh config given the problem's dimensions."""

    extra_diagnostics: tuple[str, ...] = ()
    """Names of additional diag_* flags to enable on the config."""

    repair_strategy: RepairStrategy | None = None
    """Population-level repair policy. Only applicable to evolutionary
    algorithms; ignored for L-BFGS-B. ``None`` keeps the optimizer's
    own default (``LamarckianRepair`` for every evolutionary algorithm
    — DES, CMA-ES, and MF-CMA-ES)."""

    population_initializer: PopulationInitializer | None = None
    """How the iteration-0 population is seeded. Only applicable to
    evolutionary algorithms; ignored for L-BFGS-B. ``None`` keeps the
    optimizer's own default
    (``NormalPopulationInitializer`` for DES,
    ``MeanSigmaPopulationInitializer(sigma=config.sigma)`` for CMA-ES
    and MF-CMA-ES)."""

    line_search: LineSearchStrategy | None = None
    """Line-search strategy for L-BFGS-B; ignored for evolutionary
    algorithms. ``None`` keeps the optimizer's default
    (``MoreThuenteLineSearch``)."""

    gradient_strategy: GradientStrategy | None = None
    """Gradient-approximation strategy for L-BFGS-B; ignored for
    evolutionary algorithms. ``None`` keeps the optimizer's default
    (``CentralFD``)."""

    stopping_condition: StoppingCondition | None = None
    """When to stop. ``None`` keeps the optimizer default
    (``MaxEvaluations(10_000 * dimensions)``). Pass e.g.
    ``MaxEvaluations(5000)``, ``MaxTime(30.0)``, or a composite
    (``MaxEvaluations(5000) | TargetFitness(1e-8)``) to override."""

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
        if self.algorithm == AlgorithmChoice.LBFGSB:
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
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
            **kwargs,
        ).optimize()

        return self.trace_from_result(problem, seed, result)


class HandoffAlgorithm(BenchmarkAlgorithm):
    """Two-phase (warm-up -> refinement) handoff base class.

    Saves the trace-stitching boilerplate every handoff would otherwise
    re-implement. Subclasses provide ``name`` + ``color`` (typically via
    :py:func:`dataclasses.dataclass`) and override :py:meth:`run_phases`
    to return the two phase results. The base class fills in a
    :class:`RunTrace` with:

    - eval counts in the refinement segment offset by the warm-up's
      total evaluations, so the convergence trace is continuous;
    - the refinement segment's fitness clamped to never exceed the
      warm-up's best (the refinement logger reports its own best, which
      starts at ``f(x0_refinement)`` — that may be slightly worse than
      the warm-up's running best);
    - ``handoff_eval`` and ``handoff_iter`` set from the warm-up totals,
      which the plotter uses to draw vertical handoff markers.

    Minimal usage::

        @dataclass
        class MyHandoff(HandoffAlgorithm):
            name: str
            color: str
            # ... whatever config fields you need ...

            def run_phases(self, problem, x0, seed):
                warmup_result   = ...  # run phase 1
                refinement_result = ...  # run phase 2, using warmup state
                return warmup_result, refinement_result

    Pass state from warm-up to refinement through ordinary local
    variables in :py:meth:`run_phases` — both phases live in the same
    method so any state from one is in scope for the other.

    For more elaborate runners (3+ phases, restarts, conditional
    branches), subclass :class:`BenchmarkAlgorithm` directly instead.
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
        # warm-up's best — its first point is f(x0_refinement) which can
        # be slightly worse than the warm-up's running minimum.
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

    1. Run CMA-ES on the problem until :attr:`cmaes_stopping_condition`
       fires (default: ``MaxEvaluations(10_000 * dimensions)``). Same seed
       and same ``x0`` as a standalone CMA-ES run will produce an identical
       CMA-ES prefix in the convergence trace.
    2. Read the cached eigendecomposition ``(B, D)`` from CMA-ES.
    3. Compose the initial Hessian for L-BFGS-B according to
       :attr:`transform` (see :class:`HandoffTransform`).
    4. Run L-BFGS-B from the CMA-ES mean with that ``B_0`` until
       :attr:`lbfgsb_stopping_condition` fires.

    Trace stitching, fitness clamping, and handoff metadata are inherited
    from :class:`HandoffAlgorithm`; this class only owns the
    CMA-ES-specific covariance transformation.
    """

    name: str
    color: str
    cmaes_config_factory: Callable[[int], CMAESConfig]
    lbfgsb_config_factory: Callable[[int], LBFGSBConfig]
    transform: HandoffTransform | str = HandoffTransform.INVERSE

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

    def _initial_hessian_from_cmaes(
        self,
        eigenvectors: NDArray[np.float64],
        eigenvalues_sqrt: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.float64] | None:
        return initial_hessian_from_cmaes(
            self.transform, eigenvectors, eigenvalues_sqrt, sigma
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

        _raw_cmaes_optimizer = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.CMAES,
            problem.function,
            x0,
            cmaes_config,
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
            eigenvectors, eigenvalues_sqrt, cmaes_optimizer.sigma,  # type: ignore[union-attr]
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
            stopping_condition=self.lbfgsb_stopping_condition,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            **lbfgsb_kwargs,
        ).optimize()

        return cmaes_result, lbfgsb_result


@dataclass
class InterleaveResult:
    """Detailed record of one :class:`InterleavedCMAESLBFGSB` run.

    :attr:`trace` is the standard :class:`RunTrace` (the overall-best
    staircase) that :class:`~src.benchmarking.Benchmark` consumes. The
    remaining fields expose the run's internal structure for the dedicated
    staircase plot (:func:`src.plotting.plot_interleaved_convergence`):

    - the CMA-ES *backbone* — best-so-far over CMA-ES generations only,
      ignoring the L-BFGS-B drops, so it stays above the overall best;
    - each L-BFGS-B burst as its own ``(evaluations, best)`` segment;
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
class InterleavedCMAESLBFGSB(BenchmarkAlgorithm):
    """Alternating CMA-ES <-> L-BFGS-B with a covariance-derived ``B_0``.

    Where :class:`CMAESLBFGSBHandoff` cuts over once, this cycles between
    the two algorithms for the whole budget:

    1. Advance CMA-ES for ``cmaes_interval`` generations, *resumed* from
       its own :class:`CMAESState` each cycle. With a shared RNG this
       reproduces a standalone CMA-ES run bit-for-bit, so the CMA-ES
       backbone is a true reference curve.
    2. Fire an L-BFGS-B *side-probe* from the current CMA-ES mean, with
       ``B_0`` derived from the CMA-ES covariance (``transform``, default
       ``C^{-1}`` — covariance only, exactly as in the one-shot handoff).
       The probe runs until it "stops advancing rapidly" — the L-BFGS-B
       ``factr`` relative-decrease test (``probe_factr``), capped by
       ``probe_max_evals``.
    3. Fold the probe's improvements into the tracked OVERALL BEST, then
       return to step 1 with CMA-ES **untouched**. The probe never feeds
       back into the CMA-ES distribution — it is a pure refinement of the
       running best.

    The result is the characteristic staircase: CMA-ES descends gently
    (the backbone) while each probe drops the overall best sharply toward
    the local minimum of the region CMA-ES currently occupies. Early
    drops are shallow (CMA-ES's covariance is still a poor Hessian model);
    later ones deepen as ``C^{-1}`` becomes an accurate model of the
    landscape.

    This is implemented directly on :class:`BenchmarkAlgorithm` rather
    than :class:`HandoffAlgorithm` (which is strictly two-phase) — exactly
    the multi-phase case ``docs/framework_design.md`` defers to a direct
    :py:meth:`run`.
    """

    name: str
    color: str
    cmaes_config_factory: Callable[[int], CMAESConfig]
    lbfgsb_config_factory: Callable[[int], LBFGSBConfig]

    cmaes_interval: int = 20
    """CMA-ES generations to run between consecutive L-BFGS-B probes (the
    handoff interval ``N``)."""

    total_budget: int = 0
    """Shared evaluation budget for the whole run. ``0`` -> use the CMA-ES
    config factory's own budget."""

    transform: HandoffTransform | str = HandoffTransform.INVERSE
    """How each probe turns the CMA-ES covariance into its ``B_0``.
    Default ``INVERSE`` (``C^{-1}``) passes covariance information only,
    matching :class:`CMAESLBFGSBHandoff`."""

    probe_factr: float = 1e7
    """L-BFGS-B relative-decrease stop for each probe: the burst hands back
    once ``(f_old - f_new)/max(|f_old|,|f_new|,1) <= probe_factr * eps`` —
    i.e. once the fast plunge flattens out. Larger -> the burst bails
    sooner (shallower steps); smaller -> it grinds closer to the local
    minimum (deeper steps)."""

    probe_pgtol: float = 1e-8
    """Projected-gradient stop for each probe (local-minimum safety)."""

    probe_max_evals: int = 1000
    """Hard cap on evaluations per probe, so a single burst cannot consume
    the whole remaining budget."""

    cmaes_extra_diagnostics: tuple[str, ...] = ()
    lbfgsb_extra_diagnostics: tuple[str, ...] = ()

    lbfgsb_line_search: LineSearchStrategy | None = None
    """Line-search strategy for the probes. ``None`` keeps the L-BFGS-B
    default (``MoreThuenteLineSearch``); ``ArmijoBacktracking`` is more
    robust on rippled / multimodal landscapes."""

    lbfgsb_gradient_strategy: GradientStrategy | None = None

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
        rng = np.random.default_rng(seed)
        dimensions = problem.dimensions

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

        while cumulative < total_budget:
            # ---- CMA-ES slice, resumed from its own state --------------
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

            # A non-budget termination means CMA-ES genuinely converged or
            # stalled; do one last probe then stop, otherwise the loop
            # would spin re-hitting the same criterion every cycle.
            if not cmaes_result.message.startswith("Maximum function evaluations"):
                cmaes_converged = True

            if cumulative >= total_budget:
                break

            # ---- L-BFGS-B side-probe from the current CMA-ES mean ------
            probe_budget = min(self.probe_max_evals, total_budget - cumulative)
            if probe_budget <= 0:
                break

            eigenvectors, eigenvalues_sqrt = cmaes.get_eigendecomposition()
            initial_hessian = initial_hessian_from_cmaes(
                self.transform, eigenvectors, eigenvalues_sqrt, cmaes.sigma
            )

            lbfgsb_config = self.lbfgsb_config_factory(dimensions)
            lbfgsb_config.initial_hessian = initial_hessian
            lbfgsb_config.factr = self.probe_factr
            lbfgsb_config.pgtol = self.probe_pgtol
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

            probe = LBFGSBOptimizer(
                problem.function,
                cmaes.mean,
                lbfgsb_config,
                stopping_condition=MaxEvaluations(probe_budget),
                lower_bounds=problem.lower_bound,
                upper_bounds=problem.upper_bound,
                **lbfgsb_kwargs,
            )
            probe_result = probe.optimize()

            burst_starts.append(cumulative)
            # Start the burst segment at the current overall-best level so
            # the drop renders as a clean step; the values are the running
            # best within the probe (monotone), which coincides with the
            # overall-best staircase wherever the probe improves on it.
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
