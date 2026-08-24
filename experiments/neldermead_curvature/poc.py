"""Proof of concept: can Nelder-Mead be made to *use* a donated Hessian?

No CMA-ES here on purpose -- the objective is to isolate the algorithmic
question ("can NM consume curvature at all?") from the estimation question
("is ``C`` a good Hessian?").  The curvature is donated directly: a
central-difference Hessian at ``x0``, SPD-regularised.  Its ``2n^2 + 2n + 1``
evaluations are **charged to the budget** of every arm that uses it.

Contenders (identical ``x0``, identical total budget)::

    NM                  no curvature at all (control)
    NM + H-simplex      initial simplex shaped by H^-1's principal axes
                        (the status-quo mechanism, given the *true* Hessian)
    NM-HC               the model step (new)
    NM-HC + scale fit   the model step, curvature shape only, magnitude fitted
    L-BFGS-B + H        reference: a gradient method handed the same matrix

Figures written to ``plots/neldermead_curvature/``::

    convergence.png     median + IQR per problem
    final_fitness.png   final-fitness distributions
    mechanism.png       why it works: acceptance, geometry, trust region
    curvature_quality.png   how good must the donated Hessian be?
    scale_robustness.png    what if only its *shape* is right?
    landscape_2d.png    a 2-D dissection of the two trajectories

Usage::

    PYTHONPATH=. uv run python experiments/neldermead_curvature/poc.py
    PYTHONPATH=. uv run python experiments/neldermead_curvature/poc.py \\
        --dim 10 --seeds 21 --budget 2000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.algorithms.lbfgsb.lbfgsb_optimizer import LBFGSBOptimizer
from declivity.benchmarking import Problem, RunTrace
from declivity.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from declivity.utils.benchmark_functions import (
    Ellipsoid,
    RippledEllipsoid,
    Rosenbrock,
    RotatedFunction,
)
from declivity.utils.constraint_handlers import BoxConstraintHandler, BoxStrategy
from declivity.utils.hessian import numerical_hessian, spd_regularize
from declivity.utils.initial_geometry import InitialGeometry
from declivity.utils.stopping_conditions import MaxEvaluations
from experiments.neldermead_curvature.hessian_completed import (
    Curvature,
    HessianCompletedNelderMead,
    NMResult,
)

OUTPUT_DIR = Path("plots/neldermead_curvature")

NM = "NM"
NM_SIMPLEX = "NM + H-simplex"
NM_HC = "NM-HC (model step)"
NM_HC_FIT = "NM-HC + scale fit"
LBFGSB = "L-BFGS-B + H"

COLORS = {
    NM: "#9aa0a6",
    NM_SIMPLEX: "#f5a623",
    NM_HC: "#39c0ed",
    NM_HC_FIT: "#37d67a",
    LBFGSB: "#e5484d",
}
ORDER = [NM, NM_SIMPLEX, NM_HC, NM_HC_FIT, LBFGSB]
USES_HESSIAN = {NM_SIMPLEX, NM_HC, NM_HC_FIT, LBFGSB}

FLOOR = 1e-32


@dataclass
class Contender:
    """Minimal ``AlgorithmRun``-shaped record: the plotters read name + color."""

    name: str
    color: str

    def run(self, problem: Problem, x0: NDArray[np.float64], seed: int) -> RunTrace:
        raise NotImplementedError("This study drives its own runner loop.")


def apply_dark_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#111111",
            "axes.facecolor": "#111111",
            "savefig.facecolor": "#111111",
            "grid.alpha": 0.25,
            "legend.framealpha": 0.35,
        }
    )


def build_problems(dim: int, rotation_seed: int = 7) -> list[Problem]:
    """A difficulty ramp in how well a fixed Hessian describes the landscape:
    exact quadratic, exact quadratic + coupling, quadratic trend + ripples,
    curved valley (where the curvature at ``x0`` is simply stale)."""
    return [
        Problem.from_benchmark(f"Ellipsoid {dim}D", Ellipsoid(dim)),
        Problem.from_benchmark(
            f"Rotated Ellipsoid {dim}D",
            RotatedFunction(Ellipsoid(dim), rotation="random", seed=rotation_seed),
        ),
        Problem.from_benchmark(f"Rippled Ellipsoid {dim}D", RippledEllipsoid(dim)),
        Problem.from_benchmark(f"Rosenbrock {dim}D", Rosenbrock(dim)),
    ]


def hessian_cost(dim: int) -> int:
    """Evaluations a central-difference Hessian costs; charged to every arm
    that consumes one."""
    return 2 * dim * dim + 2 * dim + 1


def sample_x0(problem: Problem, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(10_000 + seed)
    return rng.uniform(
        problem.lower_bound, problem.upper_bound, size=problem.dimensions
    )


def donate_curvature(problem: Problem, x0: NDArray[np.float64]) -> Curvature:
    return Curvature.from_hessian(
        spd_regularize(numerical_hessian(problem.function, x0))
    )


def eigen_power(curvature: Curvature, beta: float) -> Curvature:
    """Interpolate the donated shape toward the identity: ``lambda -> lambda^beta``.

    ``beta = 0`` is no curvature information at all, ``beta = 1`` the full
    donated Hessian, and the path between them is the natural (geodesic) one
    on the SPD cone.  A single knob for "how much of the curvature has been
    learned" -- exactly the axis a partially converged CMA-ES moves along.
    """
    eigenvalues, basis = np.linalg.eigh(curvature.shape)
    eigenvalues = np.maximum(eigenvalues, 1e-30) ** float(beta)
    matrix = (basis * eigenvalues) @ basis.T
    return Curvature.from_hessian(curvature.scale * matrix)


def rescale(curvature: Curvature, factor: float) -> Curvature:
    """Same shape, magnitude multiplied by ``factor`` -- a learned geometry that
    pins the anisotropy down but not the overall size."""
    return Curvature(
        shape=curvature.shape,
        shape_inverse=curvature.shape_inverse,
        scale=curvature.scale * float(factor),
    )


def hessian_simplex(
    curvature: Curvature, x0: NDArray[np.float64], base_size: float
) -> NDArray[np.float64]:
    """Initial simplex with edges along ``H^-1``'s principal axes.

    The same construction as the framework's ``CovarianceSimplexInitializer``:
    unit eigenvectors scaled by the *relative* per-axis standard deviations,
    absolute size decoupled into ``base_size``, so this arm differs from plain
    NM only in the simplex's *shape*.
    """
    dim = x0.size
    geometry = InitialGeometry.from_curvature(curvature.scale * curvature.shape, dim)
    steps = geometry.axis_steps(base_size=base_size, normalize=True, ratio_floor=1e-3)
    simplex = np.empty((dim + 1, dim))
    simplex[0] = x0
    simplex[1:] = x0 + steps.T
    return simplex


def base_simplex_size(x0: NDArray[np.float64]) -> float:
    """Longest simplex edge, matched to SciPy's 5 %-per-coordinate default so
    the shaped and unshaped simplices are the same size."""
    return max(0.05 * float(np.max(np.abs(x0))), 1e-8)


def to_trace(
    result: NMResult, offset: int, algorithm: str, problem: str, seed: int
) -> RunTrace:
    evaluations = [offset + int(e) for e in result.trace_evaluations]
    return RunTrace(
        algorithm=algorithm,
        problem=problem,
        seed=seed,
        evaluations=evaluations,
        best_fitness=[float(f) for f in result.trace_best],
        final_evaluations=evaluations[-1] if evaluations else offset,
        final_fitness=float(result.best_fitness),
    )


def run_nm(
    problem: Problem,
    x0: NDArray[np.float64],
    budget: int,
    curvature: Curvature | None,
    model_step: bool,
    fit_scale: bool = False,
    simplex: NDArray[np.float64] | None = None,
    record_simplices: bool = False,
) -> NMResult:
    return HessianCompletedNelderMead(
        problem.function,
        x0,
        lower_bounds=np.full(problem.dimensions, problem.lower_bound),
        upper_bounds=np.full(problem.dimensions, problem.upper_bound),
        max_evaluations=budget,
        initial_simplex=simplex,
        curvature=curvature if model_step else None,
        model_step=model_step,
        fit_scale=fit_scale,
        xatol=1e-12,
        fatol=1e-14,
        record_simplices=record_simplices,
    ).optimize()


def run_lbfgsb(
    problem: Problem, x0: NDArray[np.float64], curvature: Curvature, budget: int
) -> tuple[list[int], list[float], float]:
    geometry = InitialGeometry.from_curvature(
        curvature.scale * curvature.shape, problem.dimensions
    )
    result = LBFGSBOptimizer(
        problem.function,
        x0,
        LBFGSBConfig(dimensions=problem.dimensions),
        constraint_handler=BoxConstraintHandler(
            BoxStrategy.CLAMP,
            np.full(problem.dimensions, problem.lower_bound),
            np.full(problem.dimensions, problem.upper_bound),
        ),
        stopping_condition=MaxEvaluations(budget),
        lower_bounds=problem.lower_bound,
        upper_bounds=problem.upper_bound,
        initial_geometry=geometry,
    ).optimize()
    return (
        [int(e) for e in result.diagnostic.evaluations],
        [float(f) for f in result.diagnostic.best_fitness],
        float(result.best_fitness),
    )


# Main comparison


def run_cell(
    problem: Problem, seed: int, budget: int
) -> tuple[list[RunTrace], dict[str, NMResult]]:
    """All contenders for one (problem, seed)."""
    x0 = sample_x0(problem, seed)
    curvature = donate_curvature(problem, x0)
    cost = hessian_cost(problem.dimensions)
    informed_budget = budget - cost
    shaped = hessian_simplex(curvature, x0, base_simplex_size(x0))

    traces: list[RunTrace] = []
    diagnostics: dict[str, NMResult] = {}

    arms = [
        (NM, budget, 0, None, False, False, None),
        (NM_SIMPLEX, informed_budget, cost, None, False, False, shaped),
        (NM_HC, informed_budget, cost, curvature, True, False, None),
        (NM_HC_FIT, informed_budget, cost, curvature, True, True, None),
    ]
    for name, arm_budget, offset, arm_curvature, model_step, fit, simplex in arms:
        result = run_nm(
            problem, x0, arm_budget, arm_curvature, model_step, fit, simplex
        )
        traces.append(to_trace(result, offset, name, problem.name, seed))
        diagnostics[name] = result

    evaluations, best, final = run_lbfgsb(problem, x0, curvature, informed_budget)
    traces.append(
        RunTrace(
            algorithm=LBFGSB,
            problem=problem.name,
            seed=seed,
            evaluations=[cost + e for e in evaluations],
            best_fitness=best,
            final_evaluations=cost + (evaluations[-1] if evaluations else 0),
            final_fitness=final,
        )
    )
    return traces, diagnostics


def run_study(
    problems: list[Problem], seeds: list[int], budget: int, jobs: int
) -> tuple[
    dict[tuple[str, str], list[RunTrace]], dict[tuple[str, str], list[NMResult]]
]:
    traces: dict[tuple[str, str], list[RunTrace]] = {}
    diagnostics: dict[tuple[str, str], list[NMResult]] = {}
    for problem in problems:
        cells = Parallel(n_jobs=jobs)(
            delayed(run_cell)(problem, seed, budget) for seed in seeds
        )
        for cell_traces, cell_diagnostics in cells:  # type: ignore[union-attr]
            for trace in cell_traces:
                traces.setdefault((problem.name, trace.algorithm), []).append(trace)
            for name, result in cell_diagnostics.items():
                diagnostics.setdefault((problem.name, name), []).append(result)
        print(f"  done: {problem.name}")
    return traces, diagnostics


# Sweeps


BASELINE = "__baseline__"
"""Sentinel key for the no-curvature control inside a sweep cell.  A float
sentinel will not do: joblib pickles each cell back from a worker process, and
``nan`` keys stop comparing equal once they are no longer the same object."""


def sweep_cell(
    problem: Problem,
    seed: int,
    budget: int,
    kind: str,
    values: list[float],
) -> dict[tuple[float | str, str], float]:
    """One (problem, seed) across a curvature-quality or curvature-scale sweep."""
    x0 = sample_x0(problem, seed)
    curvature = donate_curvature(problem, x0)
    cost = hessian_cost(problem.dimensions)
    informed = budget - cost
    size = base_simplex_size(x0)

    out: dict[tuple[float | str, str], float] = {}
    for value in values:
        perturbed = (
            eigen_power(curvature, value)
            if kind == "quality"
            else rescale(curvature, value)
        )
        out[(value, NM_SIMPLEX)] = run_nm(
            problem,
            x0,
            informed,
            None,
            False,
            simplex=hessian_simplex(perturbed, x0, size),
        ).best_fitness
        out[(value, NM_HC)] = run_nm(
            problem, x0, informed, perturbed, True
        ).best_fitness
        out[(value, NM_HC_FIT)] = run_nm(
            problem, x0, informed, perturbed, True, fit_scale=True
        ).best_fitness
    out[(BASELINE, NM)] = run_nm(problem, x0, budget, None, False).best_fitness
    return out


def figure_sweep(
    problems: list[Problem],
    seeds: list[int],
    budget: int,
    kind: str,
    values: list[float],
    xlabel: str,
    title: str,
    save_path: Path,
    jobs: int,
    log_x: bool = False,
) -> None:
    fig, axes = plt.subplots(1, len(problems), figsize=(4.9 * len(problems), 4.6))
    axes = np.atleast_1d(axes)

    for ax, problem in zip(axes, problems):
        cells = Parallel(n_jobs=jobs)(
            delayed(sweep_cell)(problem, seed, budget, kind, values) for seed in seeds
        )
        for name in (NM_SIMPLEX, NM_HC, NM_HC_FIT):
            medians = [
                np.median(
                    [max(cell[(value, name)], FLOOR) for cell in cells]  # type: ignore[index]
                )
                for value in values
            ]
            ax.plot(values, medians, "o-", color=COLORS[name], lw=2.4, ms=5, label=name)
        baseline = np.median(
            [max(cell[(BASELINE, NM)], FLOOR) for cell in cells]  # type: ignore[index]
        )
        ax.axhline(
            baseline, color=COLORS[NM], lw=2.0, ls="--", label=f"{NM} (no curvature)"
        )
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.set_title(problem.name, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("median final fitness")
        ax.grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {save_path.name}")


# Mechanism figure


def figure_mechanism(
    problems: list[Problem],
    diagnostics: dict[tuple[str, str], list[NMResult]],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(3, len(problems), figsize=(4.9 * len(problems), 11.0))
    axes = np.atleast_2d(axes)

    for column, problem in enumerate(problems):
        # Row 0: how often the model step actually improves the incumbent.
        ax = axes[0, column]
        for name in (NM_HC, NM_HC_FIT):
            runs = diagnostics.get((problem.name, name), [])
            if not runs:
                continue
            grid = np.linspace(0, max(r.evaluations for r in runs), 60)
            curves = []
            for run in runs:
                if not run.attempt_evaluation:
                    curves.append(np.zeros_like(grid))
                    continue
                curves.append(
                    np.interp(
                        grid,
                        np.asarray(run.attempt_evaluation, dtype=float),
                        np.cumsum(np.asarray(run.attempt_improved_best, dtype=float)),
                        left=0.0,
                    )
                )
            ax.plot(
                grid,
                np.median(np.vstack(curves), axis=0),
                color=COLORS[name],
                lw=2.5,
                label=name,
            )
        ax.set_title(problem.name, fontsize=11)
        ax.set_xlabel("evaluations")
        ax.set_ylabel("model steps that improved\nthe incumbent (cumulative)")
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(fontsize=8)

        # Row 1: simplex *shape* quality.  Raw volume is confounded by
        # progress -- a simplex sitting on the optimum is legitimately tiny --
        # so this uses the scale-invariant Hadamard ratio instead.
        ax = axes[1, column]
        _band(
            ax,
            diagnostics,
            problem.name,
            (NM, NM_SIMPLEX, NM_HC),
            "log_shape_quality",
        )
        ax.set_xlabel("evaluations")
        ax.set_ylabel("log10 simplex shape quality\n(0 = orthogonal edges)")
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(fontsize=8)

        # Row 2: trust-region ratio — the quantity that gates the schedule.
        ax = axes[2, column]
        for name in (NM_HC, NM_HC_FIT):
            runs = diagnostics.get((problem.name, name), [])
            ratios = np.concatenate(
                [np.asarray(r.attempt_rho, dtype=float) for r in runs if r.attempt_rho]
                or [np.array([np.nan])]
            )
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size:
                ax.hist(
                    np.clip(ratios, -1.0, 2.0),
                    bins=40,
                    color=COLORS[name],
                    alpha=0.55,
                    label=name,
                )
        ax.axvline(0.25, color="#ffffff", ls="--", lw=1.4, label="schedule threshold")
        ax.set_xlabel("trust-region ratio  (actual / predicted decrease)")
        ax.set_ylabel("model-step attempts")
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Mechanism: the model step fires where the curvature is predictive, "
        "and stands down where it is not",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {save_path.name}")


def _band(ax, diagnostics, problem_name: str, names, field: str) -> None:
    for name in names:
        runs = diagnostics.get((problem_name, name), [])
        if not runs:
            continue
        grid = np.linspace(0, max(r.evaluations for r in runs), 120)
        stack = np.vstack(
            [
                np.interp(
                    grid,
                    np.asarray(run.trace_evaluations, dtype=float),
                    np.asarray(getattr(run, field), dtype=float),
                )
                for run in runs
            ]
        )
        ax.plot(grid, np.median(stack, axis=0), color=COLORS[name], lw=2.5, label=name)
        ax.fill_between(
            grid,
            np.percentile(stack, 25, axis=0),
            np.percentile(stack, 75, axis=0),
            color=COLORS[name],
            alpha=0.15,
        )


def figure_landscape_2d(save_path: Path) -> None:
    """A 2-D dissection: where the model steps actually go."""
    function = RotatedFunction(Ellipsoid(2), rotation="random", seed=3)
    problem = Problem.from_benchmark("Rotated Ellipsoid 2D", function)
    x0 = sample_x0(problem, seed=1)
    curvature = donate_curvature(problem, x0)

    runs = {
        name: run_nm(problem, x0, 240, curvature, model_step, record_simplices=True)
        for name, model_step in ((NM, False), (NM_HC, True))
    }

    span = min(float(np.max(np.abs(x0))) * 1.35 + 1.0, problem.upper_bound)
    grid = np.linspace(-span, span, 360)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.array(
        [function(np.array([a, b])) for a, b in zip(xx.ravel(), yy.ravel())]
    ).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, name in zip(axes, (NM, NM_HC)):
        run = runs[name]
        ax.contourf(
            xx, yy, np.log10(np.maximum(zz, 1e-12)), levels=45, cmap="magma", alpha=0.8
        )
        path = np.array([simplex[0] for simplex in run.simplex_history])
        ax.plot(
            path[:, 0],
            path[:, 1],
            color="#ffffff",
            lw=1.3,
            alpha=0.95,
            label="best vertex",
        )
        count = len(run.simplex_history)
        for index in (0, count // 4, count // 2, count - 1):
            simplex = run.simplex_history[index]
            polygon = np.vstack([simplex, simplex[0]])
            ax.plot(polygon[:, 0], polygon[:, 1], color="#39c0ed", lw=1.5, alpha=0.85)
        if run.model_points:
            points = np.array(run.model_points)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=34,
                color="#37d67a",
                edgecolor="#0b3d24",
                zorder=5,
                label="model steps",
            )
        ax.scatter(
            [0], [0], marker="*", s=240, color="#f5a623", zorder=6, label="optimum"
        )
        ax.set_title(
            f"{name}  —  f = {max(run.best_fitness, FLOOR):.2e} "
            f"after {run.evaluations} evaluations",
            fontsize=11,
        )
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "2-D rotated ellipsoid: Nelder-Mead crawls down the valley; "
        "the model step jumps it",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {save_path.name}")


def print_summary(
    problems: list[Problem],
    traces: dict[tuple[str, str], list[RunTrace]],
    diagnostics: dict[tuple[str, str], list[NMResult]],
) -> None:
    print("\nMedian final fitness (lower is better)\n")
    header = f"{'problem':<26}" + "".join(f"{name:>21}" for name in ORDER)
    print(header)
    print("-" * len(header))
    for problem in problems:
        row = f"{problem.name:<26}"
        for name in ORDER:
            runs = traces.get((problem.name, name), [])
            row += (
                f"{np.median([r.final_fitness for r in runs]):>21.4e}"
                if runs
                else f"{'-':>21}"
            )
        print(row)

    print(
        "\nRuns stopped by the internal convergence test rather than the budget"
        "\n(for NM-HC that is success -- machine precision; for NM + H-simplex"
        "\n it is premature stagnation, see the median fitness above)\n"
    )
    header = f"{'problem':<26}" + "".join(f"{name:>21}" for name in ORDER[:4])
    print(header)
    print("-" * len(header))
    for problem in problems:
        row = f"{problem.name:<26}"
        for name in ORDER[:4]:
            runs = diagnostics.get((problem.name, name), [])
            if not runs:
                row += f"{'-':>21}"
                continue
            stalled = sum(r.message.startswith("Converged") for r in runs)
            row += f"{f'{stalled}/{len(runs)}':>21}"
        print(row)

    print("\nModel-step telemetry (median per run)\n")
    header = (
        f"{'problem':<26}{'arm':>21}{'attempts':>10}{'accepted':>10}"
        f"{'improved':>10}{'% budget':>10}"
    )
    print(header)
    print("-" * len(header))
    for problem in problems:
        for name in (NM_HC, NM_HC_FIT):
            runs = diagnostics.get((problem.name, name), [])
            if not runs:
                continue
            attempts = np.median([r.attempts for r in runs])
            print(
                f"{problem.name:<26}{name:>21}{attempts:>10.0f}"
                f"{np.median([r.accepts for r in runs]):>10.0f}"
                f"{np.median([r.improvements for r in runs]):>10.0f}"
                f"{100 * attempts / np.median([r.evaluations for r in runs]):>10.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=21)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--sweep-seeds", type=int, default=11)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated substrings; keep only matching problems "
        "(e.g. --only Ellipsoid for the pure quadratics).",
    )
    args = parser.parse_args()

    apply_dark_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problems = build_problems(args.dim)
    if args.only:
        patterns = [token.strip().lower() for token in args.only.split(",")]
        problems = [
            problem
            for problem in problems
            if any(token in problem.name.lower() for token in patterns)
        ]
        if not problems:
            raise SystemExit(f"--only {args.only!r} matched no problems")
    seeds = list(range(args.seeds))
    print(
        f"Hessian-completed Nelder-Mead PoC — d={args.dim}, {len(seeds)} seeds, "
        f"budget {args.budget} (Hessian costs {hessian_cost(args.dim)} of it)"
    )
    traces, diagnostics = run_study(problems, seeds, args.budget, args.jobs)

    contenders = [Contender(name, COLORS[name]) for name in ORDER]
    plot_benchmark_convergence(
        traces,
        problems,
        contenders,
        ncols=2,
        floor=FLOOR,
        title=(
            f"Consuming a donated Hessian in Nelder-Mead — d={args.dim}, "
            f"{len(seeds)} seeds (median + IQR)"
        ),
        save_path=args.output_dir / "convergence.png",
    )
    print("  wrote convergence.png")
    plot_benchmark_boxplot(
        traces,
        problems,
        contenders,
        ncols=2,
        floor=FLOOR,
        title=f"Final fitness at {args.budget} evaluations — d={args.dim}",
        save_path=args.output_dir / "final_fitness.png",
    )
    print("  wrote final_fitness.png")
    figure_mechanism(problems, diagnostics, args.output_dir / "mechanism.png")
    figure_landscape_2d(args.output_dir / "landscape_2d.png")

    if not args.skip_sweeps:
        sweep_seeds = list(range(args.sweep_seeds))
        figure_sweep(
            problems,
            sweep_seeds,
            args.budget,
            "quality",
            [0.0, 0.25, 0.5, 0.75, 1.0],
            r"curvature knowledge  $\beta$   ($\lambda \to \lambda^{\beta}$)",
            "How good does the donated Hessian have to be? "
            r"($\beta=0$: identity, $\beta=1$: exact)",
            args.output_dir / "curvature_quality.png",
            args.jobs,
        )
        figure_sweep(
            problems,
            sweep_seeds,
            args.budget,
            "scale",
            [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3],
            r"magnitude error  $\alpha$   (donated $H = \alpha H_{\mathrm{true}}$)",
            "Right shape, wrong magnitude — the case a CMA-ES covariance hands you",
            args.output_dir / "scale_robustness.png",
            args.jobs,
            log_x=True,
        )

    print_summary(problems, traces, diagnostics)
    print(f"\nplots -> {args.output_dir}")


if __name__ == "__main__":
    main()
