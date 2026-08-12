"""Experiment 1: conditioner study for local optimizers.

The same local optimizer, started from the same point, seeded with different
conditioners: the CMA-ES covariance after k*d iterations (one CMA-ES run per
seed, snapshotted along its path), the true inverse Hessian (finite-difference
Hessian of the quadratic objective, computed once per seed), and the identity.

Per (dimension, variant) the study runs 25 seeds, each with its own starting
point and random rotation of the objective (both persisted in the setup
store), and produces per-optimizer convergence overlays.

This module exposes the ``Exp1Spec`` dataclass, the staged pipeline functions
(``run_setup_stage``, ``run_cmaes_stage``, ``run_hessian_stage``,
``run_local_stage``, ``run_plot_stage``), and the ``run()`` orchestrator that
wires them together. It has no CLI of its own; see
``experiments/conditioning/exp1_hydra.py`` for the Hydra-driven entrypoint
(single runs, local multirun, and SLURM array launches via
hydra-submitit-launcher).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from declivity.benchmarking import (
    Benchmark,
    ConditionedLocalAlgorithm,
    Problem,
    ProblemFamily,
    load_traces_json,
    snapshot_geometry,
)
from declivity.core.base_optimizer import BaseOptimizer, OptimizationResult
from declivity.plotting import plot_convergence_overlay
from declivity.utils.hessian import numerical_hessian, spd_regularize
from declivity.utils.initial_geometry import HessianScaling, InitialGeometry
from declivity.utils.stopping_conditions import (
    DEFAULT_EVALUATIONS_PER_DIMENSION,
    MaxEvaluations,
)
from experiments.conditioning.common import (
    CEC_OBJECTIVE,
    EDITIONS,
    ELLIPSOID_OBJECTIVE,
    HESSIAN_COLOR,
    IDENTITY_COLOR,
    LOCAL_CHOICES,
    LOCAL_LABELS,
    SAMPLING_SPAN,
    VARIANTS,
    anchor_traces,
    apply_dark_style,
    atomic_dump_yaml,
    atomic_save_npy,
    build_family,
    cmaes_config_factory,
    cmaes_dir,
    ensure_cmaes_path,
    ensure_setup,
    filter_seed,
    gap_traces,
    load_cmaes_path,
    local_config,
    plot_xmax,
    problem_optimum,
    ramp_colors,
    record_local_run,
    resolve_population_size,
)

plt.ioff()
plt.switch_backend("Agg")


@dataclass
class Exp1Spec:
    name: str = "cec17_f1"
    objective: str = CEC_OBJECTIVE
    edition: str = "cec2017"
    function_number: int = 1
    dimensions: tuple[int, ...] = (10, 30, 50, 100)
    num_seeds: int = 25
    variants: tuple[str, ...] = VARIANTS
    rotate: bool = True
    snapshot_ks: tuple[int, ...] = (2, 4, 8, 12, 16, 24, 32)
    cmaes_evaluations_per_dim: int = DEFAULT_EVALUATIONS_PER_DIMENSION
    include_hessian: bool = True
    include_identity: bool = True
    optimizers: tuple[str, ...] = ("lbfgsb", "bfgs", "powell", "neldermead")
    transform: str = "inverse"
    # Scaling only reinterprets the (shared) CMA-ES/Hessian matrices in the
    # local stage, so a study evaluates a whole list of scalings against one
    # set of CMA-ES runs; each lands in its own `<scaling>/` subtree.
    scalings: tuple[str, ...] = (str(HessianScaling.NONE),)
    population_factor: float = 0.0
    sigma0: float = SAMPLING_SPAN / 5.0
    local_budget_per_dim: int = 500
    num_workers: int = 1
    data_root: Path = field(default=Path("results/conditioning/exp1"))
    plot_root: Path = field(default=Path("plots/conditioning/exp1"))

    def payload(self) -> dict[str, Any]:
        return {
            "experiment": "exp1_conditioners",
            "name": self.name,
            "objective": self.objective,
            "edition": self.edition,
            "function_number": self.function_number,
            "dimensions": list(self.dimensions),
            "num_seeds": self.num_seeds,
            "variants": list(self.variants),
            "rotate": self.rotate,
            "snapshot_ks": list(self.snapshot_ks),
            "cmaes_evaluations_per_dim": self.cmaes_evaluations_per_dim,
            "include_hessian": self.include_hessian,
            "include_identity": self.include_identity,
            "optimizers": list(self.optimizers),
            "transform": self.transform,
            "scalings": list(self.scalings),
            "population_factor": self.population_factor,
            "sigma0": self.sigma0,
            "local_budget_per_dim": self.local_budget_per_dim,
        }

    def study_descriptor(self) -> dict[str, Any]:
        """Study-level config for the shared ``study.yaml``.

        Omits the sweep-axis fields (``dimensions``/``variants``): under a
        Hydra array every ``(dim, variant)`` cell carries only its own
        singleton and races to overwrite this one shared file, so recording
        them here just under-reports the study. The realized ``(dim, variant)``
        coverage is instead read back from the on-disk ``benchmarks/`` tree
        (see ``experiments/conditioning/visualize.py``).
        """
        descriptor = self.payload()
        descriptor.pop("dimensions", None)
        descriptor.pop("variants", None)
        return descriptor


def study_root(spec: Exp1Spec) -> Path:
    # rot{0,1} sits above the shared setup/cmaes/hessian artifacts because
    # rotation changes the objective (hence the CMA-ES path and Hessian);
    # scaling does not, so it nests *below* (see scaling_root/benchmark_dir).
    return spec.data_root / spec.name / f"rot{int(spec.rotate)}"


def setup_root(spec: Exp1Spec) -> Path:
    return study_root(spec) / "setup"


def hessian_dir(spec: Exp1Spec, dim: int, seed: int) -> Path:
    return study_root(spec) / "hessian" / f"d{dim:03d}" / f"seed{seed:02d}"


def scaling_root(spec: Exp1Spec, scaling: str) -> Path:
    """Per-scaling subtree holding only the local-optimizer outputs; the
    CMA-ES/Hessian artifacts above it are scaling-independent and reused."""
    return study_root(spec) / scaling


def benchmark_dir(spec: Exp1Spec, scaling: str, variant: str, dim: int) -> Path:
    return scaling_root(spec, scaling) / "benchmarks" / variant / f"d{dim:03d}"


def family_for(spec: Exp1Spec, variant: str, dim: int) -> ProblemFamily:
    if spec.objective == ELLIPSOID_OBJECTIVE:
        return build_family(
            None,
            None,
            dim,
            variant,
            spec.rotate,
            setup_root(spec),
            objective=ELLIPSOID_OBJECTIVE,
        )
    return build_family(
        EDITIONS[spec.edition],
        spec.function_number,
        dim,
        variant,
        spec.rotate,
        setup_root(spec),
        objective=CEC_OBJECTIVE,
    )


# Conditioners.


@dataclass(frozen=True)
class Conditioner:
    key: str
    label: str
    color: str
    kind: str  # "identity" | "hessian" | "covariance"
    k: int = 0


def conditioners_for(spec: Exp1Spec, dim: int) -> list[Conditioner]:
    ks = sorted(spec.snapshot_ks)
    colors = ramp_colors(len(ks))
    entries = [
        Conditioner(
            key=f"c{k:02d}",
            label=f"$C_{{{k * dim}}}$",
            color=colors[i],
            kind="covariance",
            k=k,
        )
        for i, k in enumerate(ks)
    ]
    if spec.include_hessian:
        entries.append(Conditioner("hessian", "$H^{-1}$", HESSIAN_COLOR, "hessian"))
    if spec.include_identity:
        entries.append(Conditioner("identity", "$I$", IDENTITY_COLOR, "identity"))
    return entries


def load_hessian_geometry(
    spec: Exp1Spec, dim: int, seed: int, scaling: str
) -> InitialGeometry:
    matrix = np.load(hessian_dir(spec, dim, seed) / "hessian.npy")
    return InitialGeometry.from_curvature(matrix, dim, scaling=scaling)


def load_snapshot(spec: Exp1Spec, variant: str, dim: int, seed: int, k: int):
    """The CMA-ES state after ``k * dim`` iterations for this run.

    A run that converged before ``k * dim`` has a frozen state, so the latest
    recorded snapshot is that state.
    """
    path_record = load_cmaes_path(cmaes_dir(study_root(spec), variant, dim, seed))
    snapshot = path_record.snapshot_at_or_before(k * dim)
    if snapshot is None:
        raise RuntimeError(
            f"No CMA-ES snapshot at or before iteration {k * dim} for "
            f"{variant}/d{dim}/seed{seed}."
        )
    return snapshot


def load_snapshot_geometry(
    spec: Exp1Spec, variant: str, dim: int, seed: int, k: int, scaling: str
) -> InitialGeometry:
    return snapshot_geometry(
        load_snapshot(spec, variant, dim, seed, k), spec.transform, scaling
    )


def geometry_provider(
    spec: Exp1Spec, variant: str, conditioner: Conditioner, scaling: str
):
    def provider(problem: Problem, seed: int) -> InitialGeometry:
        dim = problem.dimensions
        if conditioner.kind == "identity":
            return InitialGeometry.identity(dim)
        if conditioner.kind == "hessian":
            return load_hessian_geometry(spec, dim, seed, scaling)
        return load_snapshot_geometry(spec, variant, dim, seed, conditioner.k, scaling)

    return provider


def contender_name(optimizer_key: str, conditioner: Conditioner) -> str:
    return f"{LOCAL_LABELS[optimizer_key]} | {conditioner.label}"


def build_contenders(
    spec: Exp1Spec, variant: str, dim: int, scaling: str
) -> dict[str, list[ConditionedLocalAlgorithm]]:
    """Per optimizer key, one contender per conditioner, for one scaling."""
    budget = spec.local_budget_per_dim * dim
    contenders: dict[str, list[ConditionedLocalAlgorithm]] = {}
    for optimizer_key in spec.optimizers:
        choice = LOCAL_CHOICES[optimizer_key]
        runners = []
        for conditioner in conditioners_for(spec, dim):
            runners.append(
                ConditionedLocalAlgorithm(
                    name=contender_name(optimizer_key, conditioner),
                    color=conditioner.color,
                    algorithm=choice,
                    config_factory=lambda d, c=choice: local_config(c, d, "deep"),
                    geometry_provider=geometry_provider(
                        spec, variant, conditioner, scaling
                    ),
                    stopping_condition=MaxEvaluations(budget),
                    simplex_base_size=0.1 * SAMPLING_SPAN,
                    record=make_recorder(
                        spec, variant, optimizer_key, conditioner, scaling
                    ),
                )
            )
        contenders[optimizer_key] = runners
    return contenders


def make_recorder(
    spec: Exp1Spec,
    variant: str,
    optimizer_key: str,
    conditioner: Conditioner,
    scaling: str,
):
    def record(
        problem: Problem,
        seed: int,
        result: OptimizationResult,
        optimizer: BaseOptimizer,
    ) -> None:
        dim = problem.dimensions
        directory = (
            scaling_root(spec, scaling)
            / "local"
            / variant
            / f"d{dim:03d}"
            / optimizer_key
            / conditioner.key
            / f"seed{seed:02d}"
        )
        record_local_run(
            directory,
            result,
            optimizer,
            {
                "experiment": "exp1_conditioners",
                "study": spec.name,
                "problem": problem.name,
                "variant": variant,
                "dimensions": dim,
                "seed": seed,
                "optimizer": optimizer_key,
                "conditioner": conditioner.key,
                "conditioner_kind": conditioner.kind,
                "snapshot_iteration_requested": (
                    conditioner.k * dim if conditioner.kind == "covariance" else None
                ),
                # Differs from the requested one when CMA-ES converged sooner.
                "snapshot_iteration_used": (
                    load_snapshot(spec, variant, dim, seed, conditioner.k).iteration
                    if conditioner.kind == "covariance"
                    else None
                ),
                "transform": spec.transform,
                "scaling": scaling,
                "budget_evaluations": spec.local_budget_per_dim * dim,
                "x0_file": str(
                    setup_root(spec).relative_to(study_root(spec))
                    / f"d{dim:03d}"
                    / f"seed{seed:02d}"
                    / "x0.npy"
                ),
                "rotation_file": (
                    str(
                        setup_root(spec).relative_to(study_root(spec))
                        / f"d{dim:03d}"
                        / f"seed{seed:02d}"
                        / "rotation.npy"
                    )
                    if spec.rotate
                    else None
                ),
            },
        )

    return record


# Stages.


def run_setup_stage(spec: Exp1Spec) -> None:
    for dim in spec.dimensions:
        for seed in range(spec.num_seeds):
            ensure_setup(setup_root(spec), dim, seed, spec.rotate)
    print(f"[setup] persisted x0/rotation for d={list(spec.dimensions)}")


def run_cmaes_stage(spec: Exp1Spec, run_allowed: bool, force: bool) -> None:
    jobs = [
        (variant, dim, seed)
        for variant in spec.variants
        for dim in spec.dimensions
        for seed in range(spec.num_seeds)
    ]

    def one(variant: str, dim: int, seed: int) -> str:
        population_size = resolve_population_size(dim, spec.population_factor)
        directory = cmaes_dir(study_root(spec), variant, dim, seed)
        # record_cmaes_path is iteration-granular (see TODO.md); convert the
        # evaluation budget to whole generations here, at the call site.
        max_iterations = max(
            1, (spec.cmaes_evaluations_per_dim * dim) // population_size
        )
        ensure_cmaes_path(
            directory,
            family_for(spec, variant, dim),
            seed,
            cmaes_config_factory(population_size, spec.sigma0),
            interval=math.gcd(*spec.snapshot_ks) * dim,
            max_iterations=max_iterations,
            run_allowed=run_allowed,
            force=force,
            config_payload={
                **spec.payload(),
                "variant": variant,
                "dimensions": dim,
                "seed": seed,
                "population_size": population_size,
            },
        )
        return f"{variant}/d{dim}/seed{seed}"

    results = Parallel(n_jobs=spec.num_workers, backend="loky")(
        delayed(one)(*job) for job in jobs
    )
    print(f"[cmaes] {len(results or [])} paths ready")


def run_hessian_stage(spec: Exp1Spec, force: bool) -> None:
    if not spec.include_hessian:
        return
    jobs = [(dim, seed) for dim in spec.dimensions for seed in range(spec.num_seeds)]

    def one(dim: int, seed: int) -> str:
        directory = hessian_dir(spec, dim, seed)
        target = directory / "hessian.npy"
        if target.exists() and not force:
            return f"d{dim}/seed{seed} (cached)"
        # Variant-independent: the objective is the same function either way.
        # Both variants' array tasks for this dim reach this branch
        # concurrently, so the write below must be atomic (see
        # ``atomic_save_npy``) rather than a plain np.save.
        problem = family_for(spec, VARIANTS[0], dim).instance(seed)
        x0 = problem.starting_point(seed)
        matrix = spd_regularize(numerical_hessian(problem.function, x0))
        directory.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(target, matrix)
        atomic_dump_yaml(
            directory / "meta.yaml",
            {
                "dimensions": dim,
                "seed": seed,
                "method": "central-difference Hessian at x0, SPD-regularized",
                "evaluations": 2 * dim * dim + 2 * dim + 1,
            },
        )
        return f"d{dim}/seed{seed}"

    results = Parallel(n_jobs=spec.num_workers, backend="loky")(
        delayed(one)(*job) for job in jobs
    )
    print(f"[hessian] {len(results or [])} matrices ready")


def run_local_stage(spec: Exp1Spec) -> None:
    # Scaling is the innermost loop: every scaling reuses the same CMA-ES
    # paths and Hessians, differing only in how their matrices are rescaled
    # when the preconditioner is built.
    for scaling in spec.scalings:
        for variant in spec.variants:
            for dim in spec.dimensions:
                contenders = build_contenders(spec, variant, dim, scaling)
                algorithms = [
                    runner for runners in contenders.values() for runner in runners
                ]
                bench = Benchmark(
                    problems=[family_for(spec, variant, dim)],
                    algorithms=algorithms,  # pyright: ignore[reportArgumentType]
                    seeds=list(range(spec.num_seeds)),
                    output_dir=benchmark_dir(spec, scaling, variant, dim),
                    num_workers=spec.num_workers,
                )
                print(
                    f"\n[local] scaling={scaling} {variant} d={dim}: "
                    f"{len(algorithms)} contenders"
                )
                bench.run(verbose=True)
                bench.print_summary()


def run_plot_stage(spec: Exp1Spec, floor: float = 1e-9) -> None:
    apply_dark_style()
    rot = f"rot{int(spec.rotate)}"
    for scaling in spec.scalings:
        for variant in spec.variants:
            for dim in spec.dimensions:
                traces_path = benchmark_dir(spec, scaling, variant, dim) / "traces.json"
                if not traces_path.exists():
                    print(f"[plot] missing {traces_path}, skipping")
                    continue
                traces = load_traces_json(traces_path)
                family = family_for(spec, variant, dim)
                template = family.template
                optimum = problem_optimum(template)
                # Every contender shares x0 per seed, so anchoring each curve
                # at f(x0) makes them visibly start from one point.
                f0 = {
                    seed: max(
                        float(
                            family.instance(seed).function(
                                family.instance(seed).starting_point(seed)
                            )
                        )
                        - optimum,
                        floor,
                    )
                    for seed in range(spec.num_seeds)
                }
                shifted = anchor_traces(gap_traces(traces, optimum, floor), f0)
                contenders = build_contenders(spec, variant, dim, scaling)

                out = (
                    spec.plot_root / spec.name / rot / scaling / variant / f"d{dim:03d}"
                )
                out.mkdir(parents=True, exist_ok=True)
                for optimizer_key, runners in contenders.items():
                    label = LOCAL_LABELS[optimizer_key]
                    title = f"{template.name}, d={dim}, {variant}, {scaling}, {label}"
                    pooled = [
                        t
                        for runner in runners
                        for t in shifted.get((template.name, runner.name), [])
                    ]
                    xmax = plot_xmax(pooled)
                    plot_convergence_overlay(
                        shifted,
                        template,
                        runners,
                        title=title,
                        ylabel="$f(x_{best}) - f^*$",
                        floor=floor,
                        show_iqr=False,
                        annotate_final=False,
                        xmax=xmax,
                        save_path=out / f"{optimizer_key}_convergence.png",
                    )
                    seed0 = filter_seed(shifted, 0)
                    plot_convergence_overlay(
                        seed0,
                        template,
                        runners,
                        title=f"{title}, seed 0",
                        ylabel="$f(x_{best}) - f^*$",
                        floor=floor,
                        show_iqr=False,
                        annotate_final=False,
                        xmax=plot_xmax(
                            [
                                t
                                for runner in runners
                                for t in seed0.get((template.name, runner.name), [])
                            ]
                        ),
                        save_path=out / f"{optimizer_key}_convergence_seed0.png",
                    )
                    plt.close("all")
                print(f"[plot] scaling={scaling} {variant} d={dim} -> {out}")


def run(
    spec: Exp1Spec,
    *,
    replot: bool = False,
    run_cmaes: bool = True,
    force_cmaes: bool = False,
) -> None:
    """Run the full staged pipeline (setup -> cmaes -> hessian -> local ->
    plot) for ``spec``, or just re-render figures when ``replot`` is set.

    ``run_cmaes=False`` reuses persisted CMA-ES paths only, failing if any
    are missing (mirrors the old ``--skip-cmaes``); ``force_cmaes=True``
    re-runs CMA-ES even when a cached path exists.
    """
    root = study_root(spec)
    root.mkdir(parents=True, exist_ok=True)
    # Array tasks sharing this (name, rot) root race on study.yaml; atomic,
    # and study-level only (no per-cell sweep axes -- see study_descriptor).
    atomic_dump_yaml(root / "study.yaml", spec.study_descriptor())
    print(f"Study root: {root}")

    if not replot:
        run_setup_stage(spec)
        run_cmaes_stage(spec, run_allowed=run_cmaes, force=force_cmaes)
        run_hessian_stage(spec, force=False)
        run_local_stage(spec)
    run_plot_stage(spec)
