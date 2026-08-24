"""Experiment 2: full hybrid: local optimizers spun off a CMA-ES path.

One CMA-ES run per (function, seed) is recorded and persisted with state
snapshots at every ``granularity * d`` iterations.  Each local optimizer is
then launched once per snapshot (from the CMA-ES mean, seeded with the
covariance-derived geometry), and hybrid contenders "CMA+<opt>, k" are
composed *offline* for every switch interval ``k * d``: the CMA-ES timeline
with every k-aligned probe's evaluations spliced in, running-min over both.
All k values therefore share one CMA-ES run and one probe set per seed.

Outputs per (dimension, function, optimizer): composed convergence overlays
(median across seeds, secondary CMA-ES-iteration axis).  Per (dimension,
optimizer): the suite-aggregated ECDF over all functions.

Full scale (CEC 2017, all 30 functions, d in {10, 30, 50, 100})::

    PYTHONPATH=. uv run python experiments/conditioning/exp2_hybrid.py --num-workers 8

Local demonstration::

    PYTHONPATH=. uv run python experiments/conditioning/exp2_hybrid.py --demo

Re-render figures from persisted artifacts::

    PYTHONPATH=. uv run python experiments/conditioning/exp2_hybrid.py --demo --replot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from declivity.benchmarking import (
    CMAESPath,
    CMAESSnapshot,
    ProblemFamily,
    RunTrace,
    compose_switch_trace,
    effective_geometry_norm,
    load_arrays_parquet,
    load_traces_parquet,
    run_conditioned_local,
    save_traces_parquet,
    snapshot_geometry,
)
from declivity.plotting import plot_convergence_overlay, plot_suite_ecdf
from declivity.utils.initial_geometry import HessianScaling
from declivity.utils.stopping_conditions import (
    DEFAULT_EVALUATIONS_PER_DIMENSION,
    MaxEvaluations,
)
from experiments.conditioning.common import (
    BOUNDED,
    CMAES_COLOR,
    EDITIONS,
    LOCAL_ALONE_COLOR,
    LOCAL_CHOICES,
    LOCAL_LABELS,
    SAMPLING_SPAN,
    VARIANTS,
    CurveSpec,
    anchor_traces,
    apply_dark_style,
    atomic_dump_yaml,
    build_family,
    cmaes_config_factory,
    cmaes_dir,
    ensure_cmaes_path,
    ensure_setup,
    gap_traces,
    load_cmaes_path,
    load_yaml,
    local_config_for,
    plot_xmax,
    problem_optimum,
    ramp_colors,
    record_local_run,
    resolve_population_size,
)

plt.ioff()
plt.switch_backend("Agg")


@dataclass
class Exp2Spec:
    name: str = "cec17"
    edition: str = "cec2017"
    functions: tuple[int, ...] = tuple(range(1, 31))
    dimensions: tuple[int, ...] = (10, 30, 50, 100)
    num_seeds: int = 25
    variant: str = BOUNDED
    rotate: bool = False
    ks: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    granularity: float = 0.5
    cmaes_evaluations_per_dim: int = DEFAULT_EVALUATIONS_PER_DIMENSION
    population_factor: float = 4.0
    sigma0: float = SAMPLING_SPAN / 5.0
    probe_budget_per_dim: int = 200
    optimizers: tuple[str, ...] = (
        "lbfgsb",
        "bfgs",
        "powell",
        "neldermead_control",
        "neldermead",
        "neldermead_hc",
        "neldermead_hc_shaped",
    )
    transform: str = "inverse"
    # Scaling only reinterprets the CMA-ES-derived geometry in the probe
    # stage, so a study evaluates a whole list of scalings against one
    # shared set of CMA-ES runs (see scaling_root/probe_root/benchmark_dir).
    scalings: tuple[str, ...] = (str(HessianScaling.NONE),)
    num_workers: int = 1
    data_root: Path = field(default=Path("results/conditioning/exp2"))
    plot_root: Path = field(default=Path("plots/conditioning/exp2"))

    def __post_init__(self) -> None:
        for k in self.ks:
            if abs(round(k / self.granularity) - k / self.granularity) > 1e-9:
                raise ValueError(
                    f"k={k} is not a multiple of granularity={self.granularity}."
                )

    def snapshot_interval(self, dim: int) -> int:
        return max(1, round(self.granularity * dim))

    def payload(self) -> dict[str, Any]:
        return {
            "experiment": "exp2_hybrid",
            "name": self.name,
            "edition": self.edition,
            "functions": list(self.functions),
            "dimensions": list(self.dimensions),
            "num_seeds": self.num_seeds,
            "variant": self.variant,
            "rotate": self.rotate,
            "ks": list(self.ks),
            "granularity": self.granularity,
            "cmaes_evaluations_per_dim": self.cmaes_evaluations_per_dim,
            "population_factor": self.population_factor,
            "sigma0": self.sigma0,
            "probe_budget_per_dim": self.probe_budget_per_dim,
            "optimizers": list(self.optimizers),
            "transform": self.transform,
            "scalings": list(self.scalings),
        }

    def study_descriptor(self) -> dict[str, Any]:
        """Study-level config for the shared ``study.yaml``.

        Omits ``dimensions``: under a Hydra array sweeping ``dim``, every
        cell carries only its own singleton and races to overwrite this one
        shared file (see :func:`study_root`, which doesn't nest by
        dimension), so recording it here just under-reports the study. The
        realized dimension coverage is instead read back from the on-disk
        ``local``/``benchmarks`` tree.
        """
        descriptor = self.payload()
        descriptor.pop("dimensions", None)
        return descriptor


def study_root(spec: Exp2Spec) -> Path:
    return spec.data_root / spec.name


def setup_root(spec: Exp2Spec) -> Path:
    return study_root(spec) / "setup"


def family_for(spec: Exp2Spec, dim: int, function_number: int) -> ProblemFamily:
    return build_family(
        EDITIONS[spec.edition],
        function_number,
        dim,
        spec.variant,
        spec.rotate,
        setup_root(spec),
    )


def scaling_root(spec: Exp2Spec, scaling: str) -> Path:
    """Per-scaling subtree holding only the probe outputs; the CMA-ES
    artifacts above it are scaling-independent and reused."""
    return study_root(spec) / scaling


def probe_root(
    spec: Exp2Spec,
    scaling: str,
    dim: int,
    function_number: int,
    seed: int,
    optimizer_key: str,
) -> Path:
    return (
        scaling_root(spec, scaling)
        / "local"
        / spec.variant
        / f"d{dim:03d}"
        / f"f{function_number:02d}"
        / optimizer_key
        / f"seed{seed:02d}"
    )


def benchmark_dir(spec: Exp2Spec, scaling: str, dim: int, function_number: int) -> Path:
    return (
        scaling_root(spec, scaling)
        / "benchmarks"
        / spec.variant
        / f"d{dim:03d}"
        / f"f{function_number:02d}"
    )


# Stages.


def run_setup_stage(spec: Exp2Spec) -> None:
    for dim in spec.dimensions:
        for seed in range(spec.num_seeds):
            ensure_setup(setup_root(spec), dim, seed, spec.rotate)


def run_cmaes_stage(spec: Exp2Spec, run_allowed: bool, force: bool) -> None:
    jobs = [
        (dim, function_number, seed)
        for dim in spec.dimensions
        for function_number in spec.functions
        for seed in range(spec.num_seeds)
    ]

    def one(dim: int, function_number: int, seed: int) -> str:
        population_size = resolve_population_size(dim, spec.population_factor)
        max_evaluations = spec.cmaes_evaluations_per_dim * dim
        ensure_cmaes_path(
            cmaes_dir(study_root(spec), spec.variant, dim, seed, function_number),
            family_for(spec, dim, function_number),
            seed,
            cmaes_config_factory(population_size, spec.sigma0),
            interval=spec.snapshot_interval(dim),
            max_evaluations=max_evaluations,
            run_allowed=run_allowed,
            force=force,
            config_payload={
                **spec.payload(),
                "function_number": function_number,
                "dimensions": dim,
                "seed": seed,
                "population_size": population_size,
            },
        )
        return f"d{dim}/f{function_number}/seed{seed}"

    results = Parallel(n_jobs=spec.num_workers, backend="loky")(
        delayed(one)(*job) for job in jobs
    )
    print(f"[cmaes] {len(results or [])} paths ready")


def probe_simplex_base_size(snapshot: CMAESSnapshot) -> float:
    """Simplex extent matched to the current CMA-ES search scale, floored so a
    collapsed distribution cannot satisfy xatol immediately."""
    scale = float(snapshot.sigma * np.max(snapshot.eigenvalues_sqrt))
    return float(np.clip(3.0 * scale, 1e-6, 0.1 * SAMPLING_SPAN))


def run_probe_stage(spec: Exp2Spec, force: bool) -> None:
    jobs = [
        (scaling, dim, function_number, seed, optimizer_key)
        for scaling in spec.scalings
        for dim in spec.dimensions
        for function_number in spec.functions
        for seed in range(spec.num_seeds)
        for optimizer_key in spec.optimizers
    ]

    def one(
        scaling: str, dim: int, function_number: int, seed: int, optimizer_key: str
    ) -> str:
        choice = LOCAL_CHOICES[optimizer_key]
        family = family_for(spec, dim, function_number)
        problem = family.instance(seed)
        handler = problem.resolved_constraint_handler()
        path_record = load_cmaes_path(
            cmaes_dir(study_root(spec), spec.variant, dim, seed, function_number)
        )
        root = probe_root(spec, scaling, dim, function_number, seed, optimizer_key)
        probe_budget = spec.probe_budget_per_dim * dim
        base_payload = {
            "experiment": "exp2_hybrid",
            "study": spec.name,
            "problem": problem.name,
            "variant": spec.variant,
            "dimensions": dim,
            "function_number": function_number,
            "seed": seed,
            "optimizer": optimizer_key,
            "transform": spec.transform,
            "scaling": scaling,
        }
        executed = 0
        # ADAPTIVE carries the previous probe's effective magnitude into the
        # next one (see HessianScaling.ADAPTIVE); snapshots are temporally
        # ordered along this seed's CMA-ES path, so probes are processed in
        # that order and the chain is persisted per-probe (``effective_norm``
        # in config.yaml) so a cached probe still hands the value on to the
        # next uncached one when a run is resumed.
        previous_scale: float | None = None

        for snapshot in path_record.snapshots:
            directory = root / f"it{snapshot.iteration:06d}"
            if (directory / "run.parquet").exists() and not force:
                cached = load_yaml(directory / "config.yaml").get("effective_norm")
                previous_scale = None if cached is None else float(cached)
                continue
            geometry = snapshot_geometry(
                snapshot, spec.transform, scaling, prev_norm=previous_scale
            )
            result, optimizer = run_conditioned_local(
                choice,
                problem,
                snapshot.mean,
                local_config_for(optimizer_key, dim, "probe"),
                geometry,
                constraint_handler=handler,
                stopping_condition=MaxEvaluations(probe_budget),
                seed=seed,
                simplex_base_size=probe_simplex_base_size(snapshot),
            )
            effective_norm = effective_geometry_norm(choice, optimizer, geometry)
            previous_scale = (
                previous_scale if effective_norm is None else effective_norm
            )
            record_local_run(
                directory,
                result,
                optimizer,
                {
                    **base_payload,
                    "kind": "probe",
                    "snapshot_iteration": snapshot.iteration,
                    "snapshot_evaluations": snapshot.evaluations,
                    "budget_evaluations": probe_budget,
                    "effective_norm": effective_norm,
                },
            )
            executed += 1

        alone_dir = root / "alone"
        if not (alone_dir / "run.parquet").exists() or force:
            x0 = problem.starting_point(seed)
            alone_budget = path_record.trace.final_evaluations
            result, optimizer = run_conditioned_local(
                choice,
                problem,
                x0,
                local_config_for(optimizer_key, dim, "deep"),
                None,
                constraint_handler=handler,
                stopping_condition=MaxEvaluations(alone_budget),
                seed=seed,
                simplex_base_size=0.1 * SAMPLING_SPAN,
            )
            record_local_run(
                alone_dir,
                result,
                optimizer,
                {**base_payload, "kind": "alone", "budget_evaluations": alone_budget},
            )
            executed += 1
        return (
            f"{scaling}/d{dim}/f{function_number}/seed{seed}/{optimizer_key}: "
            f"{executed} runs"
        )

    results = Parallel(n_jobs=spec.num_workers, backend="loky")(
        delayed(one)(*job) for job in jobs
    )
    print(f"[probes] {len(results or [])} probe groups ready")


def load_run_trace(
    directory: Path, algorithm_name: str, problem_name: str, seed: int
) -> RunTrace:
    arrays = load_arrays_parquet(directory / "run.parquet")
    evaluations = [int(e) for e in arrays["evaluations"][0]]
    best_fitness = [float(f) for f in arrays["best_fitness"][0]]
    return RunTrace(
        algorithm=algorithm_name,
        problem=problem_name,
        seed=seed,
        evaluations=evaluations,
        best_fitness=best_fitness,
        final_evaluations=evaluations[-1] if evaluations else 0,
        final_fitness=min(best_fitness) if best_fitness else float("inf"),
    )


def hybrid_name(optimizer_key: str, k: float) -> str:
    return f"CMA+{LOCAL_LABELS[optimizer_key]}, $k={k:g}$"


def contenders_for(spec: Exp2Spec, optimizer_key: str) -> list[CurveSpec]:
    colors = ramp_colors(len(spec.ks))
    specs = [CurveSpec(LOCAL_LABELS[optimizer_key], LOCAL_ALONE_COLOR)]
    specs += [
        CurveSpec(hybrid_name(optimizer_key, k), colors[i])
        for i, k in enumerate(sorted(spec.ks))
    ]
    specs.append(CurveSpec("CMA-ES", CMAES_COLOR))
    return specs


def switch_snapshots(
    spec: Exp2Spec, path_record: CMAESPath, dim: int, k: float
) -> list[CMAESSnapshot]:
    interval = spec.snapshot_interval(dim)
    stride = round(k / spec.granularity)
    switches = []
    iteration = stride * interval
    while (snapshot := path_record.snapshot_at(iteration)) is not None:
        switches.append(snapshot)
        iteration += stride * interval
    return switches


def run_compose_stage(spec: Exp2Spec) -> None:
    for scaling in spec.scalings:
        for dim in spec.dimensions:
            for function_number in spec.functions:
                family = family_for(spec, dim, function_number)
                problem_name = family.name
                traces: dict[tuple[str, str], list[RunTrace]] = {}

                def add(name: str, trace: RunTrace) -> None:
                    traces.setdefault((problem_name, name), []).append(trace)

                for seed in range(spec.num_seeds):
                    path_record = load_cmaes_path(
                        cmaes_dir(
                            study_root(spec), spec.variant, dim, seed, function_number
                        )
                    )
                    add("CMA-ES", path_record.trace)
                    for optimizer_key in spec.optimizers:
                        root = probe_root(
                            spec, scaling, dim, function_number, seed, optimizer_key
                        )
                        add(
                            LOCAL_LABELS[optimizer_key],
                            load_run_trace(
                                root / "alone",
                                LOCAL_LABELS[optimizer_key],
                                problem_name,
                                seed,
                            ),
                        )
                        for k in spec.ks:
                            name = hybrid_name(optimizer_key, k)
                            probes = [
                                (
                                    snapshot.evaluations,
                                    load_run_trace(
                                        root / f"it{snapshot.iteration:06d}",
                                        name,
                                        problem_name,
                                        seed,
                                    ),
                                )
                                for snapshot in switch_snapshots(
                                    spec, path_record, dim, k
                                )
                            ]
                            add(
                                name,
                                compose_switch_trace(
                                    path_record.trace,
                                    probes,
                                    name,
                                    first_switch_iteration=round(k / spec.granularity)
                                    * spec.snapshot_interval(dim),
                                ),
                            )

                out = benchmark_dir(spec, scaling, dim, function_number)
                out.mkdir(parents=True, exist_ok=True)
                save_traces_parquet(traces, out / "traces.parquet")
                print(
                    f"[compose] {scaling}/d{dim}/f{function_number} -> "
                    f"{out / 'traces.parquet'}"
                )


def run_plot_stage(spec: Exp2Spec, floor: float = 1e-9) -> None:
    apply_dark_style()
    for scaling in spec.scalings:
        for dim in spec.dimensions:
            population_size = resolve_population_size(dim, spec.population_factor)
            suite_traces: dict[tuple[str, str], list[RunTrace]] = {}
            templates = []
            for function_number in spec.functions:
                traces_path = (
                    benchmark_dir(spec, scaling, dim, function_number)
                    / "traces.parquet"
                )
                if not traces_path.exists():
                    print(f"[plot] missing {traces_path}, skipping")
                    continue
                traces = load_traces_parquet(traces_path)
                suite_traces.update(traces)
                family = family_for(spec, dim, function_number)
                template = family.template
                templates.append(template)
                optimum = problem_optimum(template)
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

                out = (
                    spec.plot_root / spec.name / scaling / spec.variant / f"d{dim:03d}"
                )
                out.mkdir(parents=True, exist_ok=True)
                for optimizer_key in spec.optimizers:
                    runners = contenders_for(spec, optimizer_key)
                    pooled = [
                        trace
                        for runner in runners
                        for trace in shifted.get((template.name, runner.name), [])
                    ]
                    plot_convergence_overlay(
                        shifted,
                        template,
                        runners,
                        title=(
                            f"{template.name}, d={dim}, "
                            f"CMA-ES + {LOCAL_LABELS[optimizer_key]}"
                        ),
                        ylabel="$f(x_{best}) - f^*$",
                        floor=floor,
                        show_iqr=False,
                        annotate_final=False,
                        xmax=plot_xmax(pooled),
                        secondary_iter_lambda=population_size,
                        secondary_label="CMA-ES iterations",
                        save_path=out / f"f{function_number:02d}_{optimizer_key}.png",
                    )
                    plt.close("all")

            if not templates:
                continue
            out = spec.plot_root / spec.name / scaling / spec.variant / f"d{dim:03d}"
            for optimizer_key in spec.optimizers:
                plot_suite_ecdf(
                    suite_traces,
                    templates,
                    contenders_for(spec, optimizer_key),
                    title=(
                        f"ECDF, {spec.edition.upper()}, d={dim}, "
                        f"CMA-ES + {LOCAL_LABELS[optimizer_key]}"
                    ),
                    show_subtitle=False,
                    save_path=out / f"ecdf_{optimizer_key}.png",
                )
                plt.close("all")
            print(f"[plot] {scaling}/d={dim} -> {out}")


def parse_args() -> tuple[Exp2Spec, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true", help="Small local run: d=10, F1/F3/F5, 5 seeds."
    )
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument(
        "--edition", type=str, default="cec2017", choices=sorted(EDITIONS)
    )
    parser.add_argument("--functions", type=int, nargs="+", default=None)
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--variant", type=str, default=BOUNDED, choices=list(VARIANTS))
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument(
        "--ks",
        type=float,
        nargs="+",
        default=None,
        help="Switch intervals in units of d iterations; multiples of --granularity.",
    )
    parser.add_argument(
        "--granularity",
        type=float,
        default=None,
        help="Snapshot spacing in units of d iterations (the most granular k).",
    )
    parser.add_argument(
        "--evaluations-per-dim",
        type=int,
        default=None,
        help=(
            "CMA-ES evaluation budget: evaluations = value * d, converted to "
            "whole generations. Internal tolx/tolfun convergence can still "
            "end the run earlier."
        ),
    )
    parser.add_argument(
        "--population-factor",
        type=float,
        default=None,
        help="CMA-ES lambda = factor*d; omit for the framework default.",
    )
    parser.add_argument("--probe-budget-per-dim", type=int, default=None)
    parser.add_argument(
        "--optimizers", type=str, nargs="+", default=None, choices=sorted(LOCAL_CHOICES)
    )
    parser.add_argument(
        "--hessian-scaling",
        type=str,
        nargs="+",
        default=None,
        choices=["none", "sigma", "unit", "identity_norm", "adaptive"],
        help=(
            "Magnitude factor(s) applied on top of the inverse-covariance "
            "shape. All values share one set of CMA-ES runs and land in "
            "their own subtree. 'sigma' divides B_0 by sigma**2, "
            "reproducing the old fused sigma_inverse transform. 'adaptive' "
            "carries each seed's previous probe's effective magnitude "
            "(lbfgsb/bfgs only) into the next snapshot's probe."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--skip-cmaes",
        action="store_true",
        help="Reuse persisted CMA-ES paths only; fail if missing.",
    )
    parser.add_argument("--force-cmaes", action="store_true")
    parser.add_argument("--force-probes", action="store_true")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Figures only, from persisted composed traces.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("results/conditioning/exp2")
    )
    parser.add_argument(
        "--plot-root", type=Path, default=Path("plots/conditioning/exp2")
    )
    args = parser.parse_args()

    spec = Exp2Spec(
        edition=args.edition,
        variant=args.variant,
        rotate=args.rotate,
        num_workers=args.num_workers,
        data_root=args.data_root,
        plot_root=args.plot_root,
    )
    if args.hessian_scaling is not None:
        spec = replace(spec, scalings=tuple(args.hessian_scaling))
    if args.demo:
        spec = replace(
            spec,
            name="demo",
            functions=(1, 3, 5),
            dimensions=(10,),
            num_seeds=5,
            ks=(0.5, 1.0, 2.0, 4.0),
            cmaes_evaluations_per_dim=1_500,
            probe_budget_per_dim=150,
        )
    if args.study_name is not None:
        spec = replace(spec, name=args.study_name)
    if args.functions is not None:
        spec = replace(spec, functions=tuple(args.functions))
    if args.dims is not None:
        spec = replace(spec, dimensions=tuple(args.dims))
    if args.num_seeds is not None:
        spec = replace(spec, num_seeds=args.num_seeds)
    if args.granularity is not None:
        spec = replace(spec, granularity=args.granularity, ks=tuple(args.ks or spec.ks))
    if args.ks is not None:
        spec = replace(spec, ks=tuple(args.ks))
    if args.evaluations_per_dim is not None:
        spec = replace(spec, cmaes_evaluations_per_dim=args.evaluations_per_dim)
    if args.population_factor is not None:
        spec = replace(spec, population_factor=args.population_factor)
    if args.probe_budget_per_dim is not None:
        spec = replace(spec, probe_budget_per_dim=args.probe_budget_per_dim)
    if args.optimizers is not None:
        spec = replace(spec, optimizers=tuple(args.optimizers))
    return spec, args


def run(
    spec: Exp2Spec,
    *,
    replot: bool = False,
    run_cmaes: bool = True,
    force_cmaes: bool = False,
    force_probes: bool = False,
) -> None:
    """Run the full staged pipeline (setup -> cmaes -> probes -> compose ->
    plot) for ``spec``, or just re-render figures when ``replot`` is set.

    ``run_cmaes=False`` reuses persisted CMA-ES paths only, failing if any
    are missing (mirrors the old ``--skip-cmaes``); ``force_cmaes=True``
    re-runs CMA-ES even when a cached path exists.
    """
    root = study_root(spec)
    root.mkdir(parents=True, exist_ok=True)
    # Array tasks sharing this study name race on study.yaml when `dim` is
    # the Hydra sweep axis (study_root doesn't nest by dimension); atomic,
    # and study-level only (see study_descriptor).
    atomic_dump_yaml(root / "study.yaml", spec.study_descriptor())
    print(f"Study root: {root}")

    if not replot:
        run_setup_stage(spec)
        run_cmaes_stage(spec, run_allowed=run_cmaes, force=force_cmaes)
        run_probe_stage(spec, force=force_probes)
        run_compose_stage(spec)
    run_plot_stage(spec)


def main() -> None:
    spec, args = parse_args()
    run(
        spec,
        replot=args.replot,
        run_cmaes=not args.skip_cmaes,
        force_cmaes=args.force_cmaes,
        force_probes=args.force_probes,
    )


if __name__ == "__main__":
    main()
