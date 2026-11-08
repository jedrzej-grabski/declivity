"""Experiment 3: how the conditioner reaches Nelder-Mead's initial simplex.

A Nelder-Mead-only slice of the exp1 conditioner study, in its own artifact
store, for iterating on the simplex-seeding code without touching (or waiting
for) the four-optimizer grid.  Same conditioners, same starting points, same
25 seeds, same three scalings -- only the optimizer list is narrowed to
Nelder-Mead and the dimension list to one entry.

Why it exists separately: the geometry reaches each local optimizer through a
different accessor (``B_0`` for L-BFGS-B, ``principal_directions`` for Powell,
``axis_steps`` for Nelder-Mead), so a change to one accessor is only visible in
one optimizer's runs.  Isolating that optimizer makes the change's effect
legible and cheap to re-measure.

The simplex modes
-----------------
``InitialGeometry.axis_steps`` sizes the simplex through one of three
branches, and which one runs is what this experiment varies:

``--simplex-mode normalize`` (default)
    ``length_k = base_size * clip(D_k/max(D), ratio_floor, 1)`` -- relative
    anisotropy only, with absolute size decoupled into ``base_size``.  This is
    exactly what exp1 runs, so this mode reproduces the grid's Nelder-Mead
    panels.

``--simplex-mode absolute``
    ``length_k`` comes from the geometry's own extent, the branch guarded by
    ``absolute=True``.  ``NelderMeadOptimizer`` never sets that flag when it
    builds the initializer from an ``initial_geometry=``, so this branch is
    unreachable through the normal seam; :class:`AbsoluteSimplexNelderMead`
    below constructs the initializer explicitly to exercise it.

``--simplex-mode absolute_scaled``
    The absolute branch's ``sigma * D_k`` form, driven through the equivalent
    ``normalize=False`` branch with ``base_size=sigma``.  Run this against
    ``absolute`` to isolate the effect of the per-axis scales alone: the two
    differ only in whether ``D_k`` survives, with the branch, the sizes and
    the floor all held fixed.  Comparing ``normalize`` against ``absolute``
    confounds that question with the base-size change.

Two things to know when reading absolute-mode results.  ``sigma`` is only a
real quantity for a geometry built from a CMA-ES snapshot: ``identity()`` and
``from_curvature()`` both default it to ``1.0``, so the identity and ``H^-1``
reference conditioners get a unit-extent simplex rather than the
``0.1 * SAMPLING_SPAN`` one they get in normalize mode.  And the covariance
conditioners' extent *shrinks with k* as CMA-ES converges, which is what
``min_step`` (``100 * xatol``) exists to floor.

Usage::

    # reproduce exp1's Nelder-Mead panels at d=10
    PYTHONPATH=. uv run python -m experiments.conditioning.exp3_nm_simplex

    # exercise the absolute branch instead
    PYTHONPATH=. uv run python -m experiments.conditioning.exp3_nm_simplex \\
        --simplex-mode absolute

    # all three, into separate stores, for a side-by-side
    PYTHONPATH=. uv run python -m experiments.conditioning.exp3_nm_simplex \\
        --simplex-mode both --num-workers 12

Stages are the exp1 ones and stay idempotent: ``setup`` / ``cmaes`` /
``hessian`` are cached and shared by every mode (none of them depend on the
simplex code), so re-running after an edit only recomputes the local stage.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.neldermead.config import NelderMeadConfig
from declivity.algorithms.neldermead.neldermead_optimizer import NelderMeadOptimizer
from declivity.benchmarking import (
    Benchmark,
    ConditionedLocalAlgorithm,
    Problem,
    RunTrace,
)
from declivity.benchmarking.algorithm_run import BenchmarkAlgorithm
from declivity.core.config_base import BaseConfig
from declivity.utils.initial_geometry import InitialGeometry
from declivity.utils.population_initializers import CovarianceSimplexInitializer
from declivity.utils.stopping_conditions import MaxEvaluations, StoppingCondition
from experiments.conditioning.common import (
    ELLIPSOID_OBJECTIVE,
    OBJECTIVES,
    SAMPLING_SPAN,
    VARIANTS,
    apply_dark_style,
    local_config,
)
from experiments.conditioning.exp1_conditioners import (
    Exp1Spec,
    benchmark_dir,
    conditioners_for,
    contender_name,
    family_for,
    geometry_provider,
    run_cmaes_stage,
    run_hessian_stage,
    run_plot_stage,
    run_setup_stage,
    study_root,
)

GeometryProvider = Callable[[Problem, int], InitialGeometry | None]

plt.ioff()
plt.switch_backend("Agg")

OPTIMIZER_KEY = "neldermead"

NORMALIZE = "normalize"
ABSOLUTE = "absolute"
ABSOLUTE_SCALED = "absolute_scaled"
SIMPLEX_MODES = (NORMALIZE, ABSOLUTE, ABSOLUTE_SCALED)


def recover_sigma(geometry: InitialGeometry) -> float:
    """The geometry's ``sigma``, from public accessors only.

    ``principal_scales()`` is the covariance shape ``D``;
    ``principal_scales(include_sigma=True)`` is ``sigma * D``.  Their ratio is
    ``sigma``, which the class otherwise keeps private.  Both are ordered
    identically, so any index works; the first is the largest and therefore
    the least likely to be sitting on the eigenvalue floor.
    """
    shape = geometry.principal_scales()
    scaled = geometry.principal_scales(include_sigma=True)
    if shape.size == 0 or shape[0] <= 0.0:
        return 1.0
    return float(scaled[0] / shape[0])


@dataclass
class AbsoluteSimplexNelderMead(BenchmarkAlgorithm):
    """Nelder-Mead seeded through an explicitly built absolute-mode simplex.

    The ``initial_geometry=`` constructor seam always builds its
    ``CovarianceSimplexInitializer`` with ``absolute=False``, so the absolute
    branch of ``InitialGeometry.axis_steps`` cannot be reached that way.  This
    runner builds the initializer itself and passes it as
    ``population_initializer=``, which is the same seam the framework uses for
    every other population initializer -- no library change needed.

    Apart from the initializer this is ``ConditionedLocalAlgorithm`` with
    ``algorithm=NELDERMEAD``: same shared ``x0``, same per-run geometry
    provider, so its traces drop into the same store and plot through the same
    panels.
    """

    name: str
    color: str
    config_factory: Callable[[int], BaseConfig]
    geometry_provider: GeometryProvider

    keep_axis_scales: bool = False
    """``False`` uses the ``absolute=True`` branch as it currently stands, one
    length ``sigma`` for every axis.  ``True`` reproduces the ``sigma * D_k``
    lengths that branch produced before, by driving the equivalent
    ``normalize=False`` branch with ``base_size=sigma`` — so the two settings
    differ *only* in whether the per-axis scales survive, which is what makes
    them comparable."""

    stopping_condition: StoppingCondition | None = None

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        config = self.config_factory(problem.dimensions)
        assert isinstance(config, NelderMeadConfig)
        geometry = self.geometry_provider(problem, seed)
        assert geometry is not None, "every exp1 conditioner yields a geometry"

        initializer = (
            CovarianceSimplexInitializer(
                geometry,
                base_size=recover_sigma(geometry),
                normalize=False,
                min_step=100.0 * config.xatol,
            )
            if self.keep_axis_scales
            else CovarianceSimplexInitializer(
                geometry,
                # base_size is unused on the absolute branch (the geometry's
                # own extent sets the lengths); min_step matches what the
                # initial_geometry= seam would have applied.
                min_step=100.0 * config.xatol,
                absolute=True,
            )
        )
        optimizer = NelderMeadOptimizer(
            problem.function,
            x0,
            config,
            population_initializer=initializer,
            constraint_handler=self.resolve_constraint_handler(problem),
            stopping_condition=self.stopping_condition,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
        )
        result = optimizer.optimize()
        return self.trace_from_result(problem, seed, result)


def spec_for(
    mode: str,
    dim: int,
    objective: str,
    edition: str,
    function_number: int,
    num_seeds: int,
    rotate: bool,
    variants: tuple[str, ...],
    scalings: tuple[str, ...],
    budget_per_dim: int,
    num_workers: int,
    data_root: Path,
    plot_root: Path,
    study_name: str | None,
) -> Exp1Spec:
    """An ``Exp1Spec`` narrowed to Nelder-Mead at one dimension.

    ``name`` carries the simplex mode, so the two modes get sibling stores
    under the same roots and never overwrite each other -- or exp1's.
    """
    base = study_name or (
        "ellipsoid" if objective == ELLIPSOID_OBJECTIVE else f"cec_f{function_number}"
    )
    return Exp1Spec(
        name=f"nm_{base}_{mode}",
        objective=objective,
        edition=edition,
        function_number=function_number,
        dimensions=(dim,),
        num_seeds=num_seeds,
        variants=variants,
        rotate=rotate,
        scalings=scalings,
        optimizers=(OPTIMIZER_KEY,),
        local_budget_per_dim=budget_per_dim,
        num_workers=num_workers,
        data_root=data_root,
        plot_root=plot_root,
    )


def nelder_mead_config(dim: int) -> BaseConfig:
    return local_config(AlgorithmChoice.NELDERMEAD, dim, "deep")


def build_contenders(
    spec: Exp1Spec, variant: str, dim: int, scaling: str, mode: str
) -> list[BenchmarkAlgorithm]:
    """One Nelder-Mead contender per conditioner, in the requested mode."""
    budget = spec.local_budget_per_dim * dim
    runners: list[BenchmarkAlgorithm] = []
    for conditioner in conditioners_for(spec, dim):
        name = contender_name(OPTIMIZER_KEY, conditioner)
        provider = geometry_provider(spec, variant, conditioner, scaling)
        if mode in (ABSOLUTE, ABSOLUTE_SCALED):
            runners.append(
                AbsoluteSimplexNelderMead(
                    name=name,
                    color=conditioner.color,
                    config_factory=nelder_mead_config,
                    geometry_provider=provider,
                    stopping_condition=MaxEvaluations(budget),
                    keep_axis_scales=mode == ABSOLUTE_SCALED,
                )
            )
        else:
            # The same class exp1 uses, with the same simplex_base_size, so
            # this mode is a reproduction rather than a re-implementation.
            runners.append(
                ConditionedLocalAlgorithm(
                    name=name,
                    color=conditioner.color,
                    config_factory=nelder_mead_config,
                    geometry_provider=provider,
                    stopping_condition=MaxEvaluations(budget),
                    algorithm=AlgorithmChoice.NELDERMEAD,
                    simplex_base_size=0.1 * SAMPLING_SPAN,
                )
            )
    return runners


def run_local_stage(spec: Exp1Spec, mode: str) -> None:
    for scaling in spec.scalings:
        for variant in spec.variants:
            for dim in spec.dimensions:
                runners = build_contenders(spec, variant, dim, scaling, mode)
                bench = Benchmark(
                    problems=[family_for(spec, variant, dim)],
                    algorithms=runners,  # pyright: ignore[reportArgumentType]
                    seeds=list(range(spec.num_seeds)),
                    output_dir=benchmark_dir(spec, scaling, variant, dim),
                    num_workers=spec.num_workers,
                )
                print(
                    f"\n[local] mode={mode} scaling={scaling} {variant} d={dim}: "
                    f"{len(runners)} contenders"
                )
                bench.run(verbose=True)
                bench.print_summary()


def report_simplex_extents(spec: Exp1Spec, mode: str) -> None:
    """Print the initial simplex edge lengths each conditioner produces.

    The simplex is what this experiment varies, so its geometry is worth
    showing directly rather than inferring it from convergence curves.
    """
    dim = spec.dimensions[0]
    variant, scaling, seed = spec.variants[0], spec.scalings[0], 0
    config = local_config(AlgorithmChoice.NELDERMEAD, dim, "deep")
    assert isinstance(config, NelderMeadConfig)
    problem = family_for(spec, variant, dim).instance(seed)

    print(
        f"\n[simplex] mode={mode} d={dim} {variant} scaling={scaling} seed={seed}: "
        f"edge lengths across the {dim} simplex axes"
    )
    print(f"  {'conditioner':<12} {'min':>11} {'median':>11} {'max':>11}  anisotropy")
    for conditioner in conditioners_for(spec, dim):
        geometry = geometry_provider(spec, variant, conditioner, scaling)(problem, seed)
        assert geometry is not None
        if mode == ABSOLUTE_SCALED:
            steps = geometry.axis_steps(
                base_size=recover_sigma(geometry),
                normalize=False,
                min_step=100.0 * config.xatol,
            )
        else:
            steps = geometry.axis_steps(
                base_size=0.1 * SAMPLING_SPAN,
                min_step=100.0 * config.xatol,
                absolute=mode == ABSOLUTE,
            )
        lengths = np.linalg.norm(steps, axis=0)
        ratio = float(np.max(lengths) / max(np.min(lengths), 1e-300))
        print(
            f"  {conditioner.key:<12} {np.min(lengths):11.4e} "
            f"{np.median(lengths):11.4e} {np.max(lengths):11.4e}  {ratio:9.2e}"
        )


def run(spec: Exp1Spec, mode: str, run_cmaes: bool, force_cmaes: bool) -> None:
    root = study_root(spec)
    root.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\nStudy root: {root}  (simplex mode: {mode})\n{'=' * 78}")

    run_setup_stage(spec)
    run_cmaes_stage(spec, run_allowed=run_cmaes, force=force_cmaes)
    run_hessian_stage(spec, force=False)
    report_simplex_extents(spec, mode)
    run_local_stage(spec, mode)
    # exp1's plot stage keys figures off contender *names*, which
    # build_contenders reproduces exactly, so it renders this store unchanged.
    run_plot_stage(spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--simplex-mode",
        default=NORMALIZE,
        choices=[*SIMPLEX_MODES, "both"],
        help="Which axis_steps branch to seed the simplex through "
        f"(default: {NORMALIZE}, matching exp1).",
    )
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--num-seeds", type=int, default=25)
    parser.add_argument(
        "--objective", default=ELLIPSOID_OBJECTIVE, choices=list(OBJECTIVES)
    )
    parser.add_argument("--edition", default="cec2017")
    parser.add_argument("--function-number", type=int, default=1)
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS)
    )
    parser.add_argument(
        "--scalings", nargs="+", default=["none", "unit", "identity_norm"]
    )
    parser.add_argument("--budget-per-dim", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--data-root", type=Path, default=Path("results/conditioning/exp3")
    )
    parser.add_argument(
        "--plot-root", type=Path, default=Path("plots/conditioning/exp3")
    )
    parser.add_argument("--skip-cmaes", action="store_true")
    parser.add_argument("--force-cmaes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_dark_style()
    modes = list(SIMPLEX_MODES) if args.simplex_mode == "both" else [args.simplex_mode]
    for mode in modes:
        spec = spec_for(
            mode,
            dim=args.dim,
            objective=args.objective,
            edition=args.edition,
            function_number=args.function_number,
            num_seeds=args.num_seeds,
            rotate=not args.no_rotate,
            variants=tuple(args.variants),
            scalings=tuple(args.scalings),
            budget_per_dim=args.budget_per_dim,
            num_workers=args.num_workers,
            data_root=args.data_root,
            plot_root=args.plot_root,
            study_name=args.study_name,
        )
        run(spec, mode, run_cmaes=not args.skip_cmaes, force_cmaes=args.force_cmaes)
    print(f"\nDone. Figures under {args.plot_root}/")


if __name__ == "__main__":
    main()
