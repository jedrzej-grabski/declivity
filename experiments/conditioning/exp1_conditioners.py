"""Experiment 1: conditioner study for local optimizers.

The same local optimizer, started from the same point, seeded with different
conditioners: the CMA-ES covariance after k*d iterations (one CMA-ES run per
seed, snapshotted along its path), the true inverse Hessian (finite-difference
Hessian of the quadratic objective, computed once per seed), and the identity.

Per (dimension, variant) the study runs 25 seeds, each with its own starting
point and random rotation of the objective (both persisted in the setup
store), and produces per-optimizer convergence overlays.

Full scale (CEC 2017 F1, d in {10, 30, 50, 100}, bounded + unbounded)::

    PYTHONPATH=. uv run python experiments/conditioning/exp1_conditioners.py --num-workers 8

Local demonstration::

    PYTHONPATH=. uv run python experiments/conditioning/exp1_conditioners.py --demo

Re-render figures from persisted artifacts (no optimizer runs)::

    PYTHONPATH=. uv run python experiments/conditioning/exp1_conditioners.py --demo --replot
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
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
from declivity.utils.initial_geometry import InitialGeometry
from declivity.utils.stopping_conditions import MaxEvaluations
from experiments.conditioning.common import (
    CEC_OBJECTIVE,
    EDITIONS,
    ELLIPSOID_OBJECTIVE,
    HESSIAN_COLOR,
    IDENTITY_COLOR,
    LOCAL_CHOICES,
    LOCAL_LABELS,
    OBJECTIVES,
    SAMPLING_SPAN,
    VARIANTS,
    anchor_traces,
    apply_dark_style,
    build_family,
    cmaes_config_factory,
    cmaes_dir,
    dump_yaml,
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
    include_hessian: bool = True
    include_identity: bool = True
    optimizers: tuple[str, ...] = ("lbfgsb", "bfgs", "powell", "neldermead")
    transform: str = "inverse"
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
            "include_hessian": self.include_hessian,
            "include_identity": self.include_identity,
            "optimizers": list(self.optimizers),
            "transform": self.transform,
            "population_factor": self.population_factor,
            "sigma0": self.sigma0,
            "local_budget_per_dim": self.local_budget_per_dim,
        }


def study_root(spec: Exp1Spec) -> Path:
    return spec.data_root / spec.name


def setup_root(spec: Exp1Spec) -> Path:
    return study_root(spec) / "setup"


def hessian_dir(spec: Exp1Spec, dim: int, seed: int) -> Path:
    return study_root(spec) / "hessian" / f"d{dim:03d}" / f"seed{seed:02d}"


def benchmark_dir(spec: Exp1Spec, variant: str, dim: int) -> Path:
    return study_root(spec) / "benchmarks" / variant / f"d{dim:03d}"


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


def load_hessian_geometry(spec: Exp1Spec, dim: int, seed: int) -> InitialGeometry:
    matrix = np.load(hessian_dir(spec, dim, seed) / "hessian.npy")
    return InitialGeometry.from_curvature(matrix, dim)


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
    spec: Exp1Spec, variant: str, dim: int, seed: int, k: int
) -> InitialGeometry:
    return snapshot_geometry(load_snapshot(spec, variant, dim, seed, k), spec.transform)


def geometry_provider(spec: Exp1Spec, variant: str, conditioner: Conditioner):
    def provider(problem: Problem, seed: int) -> InitialGeometry:
        dim = problem.dimensions
        if conditioner.kind == "identity":
            return InitialGeometry.identity(dim)
        if conditioner.kind == "hessian":
            return load_hessian_geometry(spec, dim, seed)
        return load_snapshot_geometry(spec, variant, dim, seed, conditioner.k)

    return provider


def contender_name(optimizer_key: str, conditioner: Conditioner) -> str:
    return f"{LOCAL_LABELS[optimizer_key]} | {conditioner.label}"


def build_contenders(
    spec: Exp1Spec, variant: str, dim: int
) -> dict[str, list[ConditionedLocalAlgorithm]]:
    """Per optimizer key, one contender per conditioner."""
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
                    geometry_provider=geometry_provider(spec, variant, conditioner),
                    stopping_condition=MaxEvaluations(budget),
                    simplex_base_size=0.1 * SAMPLING_SPAN,
                    record=make_recorder(spec, variant, optimizer_key, conditioner),
                )
            )
        contenders[optimizer_key] = runners
    return contenders


def make_recorder(
    spec: Exp1Spec, variant: str, optimizer_key: str, conditioner: Conditioner
):
    def record(
        problem: Problem,
        seed: int,
        result: OptimizationResult,
        optimizer: BaseOptimizer,
    ) -> None:
        dim = problem.dimensions
        directory = (
            study_root(spec)
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
        ensure_cmaes_path(
            directory,
            family_for(spec, variant, dim),
            seed,
            cmaes_config_factory(population_size, spec.sigma0),
            interval=math.gcd(*spec.snapshot_ks) * dim,
            max_iterations=max(spec.snapshot_ks) * dim,
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
        problem = family_for(spec, VARIANTS[0], dim).instance(seed)
        x0 = problem.starting_point(seed)
        matrix = spd_regularize(numerical_hessian(problem.function, x0))
        directory.mkdir(parents=True, exist_ok=True)
        np.save(target, matrix)
        dump_yaml(
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
    for variant in spec.variants:
        for dim in spec.dimensions:
            contenders = build_contenders(spec, variant, dim)
            algorithms = [
                runner for runners in contenders.values() for runner in runners
            ]
            bench = Benchmark(
                problems=[family_for(spec, variant, dim)],
                algorithms=algorithms,  # pyright: ignore[reportArgumentType]
                seeds=list(range(spec.num_seeds)),
                output_dir=benchmark_dir(spec, variant, dim),
                num_workers=spec.num_workers,
            )
            print(f"\n[local] {variant} d={dim}: {len(algorithms)} contenders")
            bench.run(verbose=True)
            bench.print_summary()


def run_plot_stage(spec: Exp1Spec, floor: float = 1e-9) -> None:
    apply_dark_style()
    for variant in spec.variants:
        for dim in spec.dimensions:
            traces_path = benchmark_dir(spec, variant, dim) / "traces.json"
            if not traces_path.exists():
                print(f"[plot] missing {traces_path}, skipping")
                continue
            traces = load_traces_json(traces_path)
            family = family_for(spec, variant, dim)
            template = family.template
            optimum = problem_optimum(template)
            # Every contender shares x0 per seed, so anchoring each curve at
            # f(x0) makes them visibly start from one point.
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
            contenders = build_contenders(spec, variant, dim)

            out = spec.plot_root / spec.name / variant / f"d{dim:03d}"
            out.mkdir(parents=True, exist_ok=True)
            for optimizer_key, runners in contenders.items():
                label = LOCAL_LABELS[optimizer_key]
                title = f"{template.name}, d={dim}, {variant}, {label}"
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
            print(f"[plot] {variant} d={dim} -> {out}")


def parse_args() -> tuple[Exp1Spec, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Small local run: d=10, 5 seeds, k in {1,2,4}.",
    )
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument(
        "--objective",
        type=str,
        default=CEC_OBJECTIVE,
        choices=sorted(OBJECTIVES),
        help=(
            "'cec': CEC2017 F1 (or --edition/--function), the suite's default "
            "rotated/shifted benchmark. 'ellipsoid': the axis-aligned "
            "10^6-conditioned Ellipsoid (utils/benchmark_functions.Ellipsoid), "
            "the canonical CMA-ES-covariance-converges-to-Hessian test "
            "function, with no baked-in rotation/shift of its own -- use it "
            "when you want the suite's --no-rotate toggle to be the *only* "
            "source of coordinate coupling. --edition/--function are "
            "rejected when --objective ellipsoid is selected."
        ),
    )
    parser.add_argument("--edition", type=str, default=None, choices=sorted(EDITIONS))
    parser.add_argument("--function", type=int, default=None)
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument(
        "--variants", type=str, nargs="+", default=None, choices=list(VARIANTS)
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=None,
        help="Snapshot multipliers: conditioner C after k*d CMA-ES iterations.",
    )
    parser.add_argument(
        "--optimizers", type=str, nargs="+", default=None, choices=sorted(LOCAL_CHOICES)
    )
    parser.add_argument(
        "--transform", type=str, default="inverse", choices=["inverse", "sigma_inverse"]
    )
    parser.add_argument(
        "--population-factor",
        type=float,
        default=0.0,
        help="CMA-ES lambda = factor*d; 0 = framework default.",
    )
    parser.add_argument("--local-budget-per-dim", type=int, default=None)
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--no-hessian", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--skip-cmaes",
        action="store_true",
        help="Reuse persisted CMA-ES paths only; fail if missing.",
    )
    parser.add_argument(
        "--force-cmaes", action="store_true", help="Re-run CMA-ES even when cached."
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Figures only, from persisted benchmark traces.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("results/conditioning/exp1")
    )
    parser.add_argument(
        "--plot-root", type=Path, default=Path("plots/conditioning/exp1")
    )
    args = parser.parse_args()

    if args.objective == ELLIPSOID_OBJECTIVE:
        if args.edition is not None or args.function is not None:
            parser.error(
                "--edition/--function are not applicable with --objective ellipsoid."
            )
        edition = "cec2017"  # unused placeholder; ignored by family_for()
        function_number = 1
        default_name = "ellipsoid"
    else:
        edition = args.edition or "cec2017"
        function_number = args.function if args.function is not None else 1
        default_name = "cec17_f1"

    spec = Exp1Spec(
        name=default_name,
        objective=args.objective,
        edition=edition,
        function_number=function_number,
        data_root=args.data_root,
        plot_root=args.plot_root,
        num_workers=args.num_workers,
        transform=args.transform,
        population_factor=args.population_factor,
        rotate=not args.no_rotate,
        include_hessian=not args.no_hessian,
    )
    if args.demo:
        spec = replace(
            spec,
            name="demo",
            dimensions=(10,),
            num_seeds=5,
            snapshot_ks=(2, 4, 8, 16, 32),
            local_budget_per_dim=300,
        )
    if args.study_name is not None:
        spec = replace(spec, name=args.study_name)
    if args.dims is not None:
        spec = replace(spec, dimensions=tuple(args.dims))
    if args.num_seeds is not None:
        spec = replace(spec, num_seeds=args.num_seeds)
    if args.variants is not None:
        spec = replace(spec, variants=tuple(args.variants))
    if args.ks is not None:
        spec = replace(spec, snapshot_ks=tuple(args.ks))
    if args.optimizers is not None:
        spec = replace(spec, optimizers=tuple(args.optimizers))
    if args.local_budget_per_dim is not None:
        spec = replace(spec, local_budget_per_dim=args.local_budget_per_dim)
    return spec, args


def main() -> None:
    spec, args = parse_args()
    root = study_root(spec)
    root.mkdir(parents=True, exist_ok=True)
    dump_yaml(root / "study.yaml", spec.payload())
    print(f"Study root: {root}")

    if not args.replot:
        run_setup_stage(spec)
        run_cmaes_stage(spec, run_allowed=not args.skip_cmaes, force=args.force_cmaes)
        run_hessian_stage(spec, force=False)
        run_local_stage(spec)
    run_plot_stage(spec)


if __name__ == "__main__":
    main()
