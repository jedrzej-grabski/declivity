"""Cross-validation: framework-native CMA-ES vs the historical reference port.

The framework-native :class:`~src.algorithms.cmaes.CMAESOptimizer` is a
clean rewrite that uses the framework's
:class:`~src.utils.repair_strategies.RepairStrategy` and
:class:`~src.utils.population_initializers.PopulationInitializer` seams.
In the *default* configuration the two implementations consume the RNG
stream differently (the framework's
``MeanSigmaPopulationInitializer`` draws a ``(dim, λ)`` block at
iteration 0; the reference draws ``λ`` independent ``dim``-vectors) and
the framework clamps infeasible candidates rather than rejecting them.

For this cross-validation oracle, the framework run is driven by hand
with **reference-matched sampling**:

* per-individual ``standard_normal(dim)`` calls (so the RNG sequence
  matches the reference exactly),
* up to ``n_max_resampling`` rejection attempts before clamping (matching
  ``CMA.ask``).

With those two adjustments the framework and the reference are
bit-identical at every generation — the convergence overlays sit
exactly on top of each other and the per-element max-diff heatmap is at
or near floating-point noise.  That isolates "is the framework rewrite
correct?" from "does the framework's default sampling/repair shape
behave like the reference?" (the latter is what
``experiments/cross_validation/cmaes_components.py`` covers).

Output (under ``plots/cross_validation/cmaes_vs_reference/``):

- ``convergence_<func>_d<dim>.png`` — per-function multi-seed overlay
- ``state_trajectories_<func>_d<dim>.png`` — sigma / mean-norm / C-trace
  trajectories side by side for a single seed
- ``summary.csv`` — per-(function × dim × seed) row with final fitness,
  evaluations, and the max absolute mean / C / pc differences observed
  across the full run.

Run::

    PYTHONPATH=. pdm run python experiments/cross_validation/cmaes_vs_reference.py
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from declivity.algorithms.cmaes.cmaes_optimizer import CMAESOptimizer
from declivity.algorithms.cmaes.cmaes_reference import CMA
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.utils.benchmark_functions import (
    Ackley,
    BenchmarkFunction,
    CEC17Function,
    Ellipsoid,
    Rastrigin,
    Rosenbrock,
    Sphere,
)
from declivity.utils.constraint_handlers import BoxConstraintHandler, BoxStrategy
from declivity.utils.stopping_conditions import MaxEvaluations

# ---------------------------------------------------------------------------
# Per-run records.
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One run of one algorithm — bookkeeping for plotting and summary."""

    name: str
    fn_name: str
    dim: int
    seed: int

    best_fitness: float = math.inf
    evaluations: int = 0
    wall_time: float = 0.0
    converged_message: str = ""

    # Per-iteration trace.
    iter_best: list[float] = field(default_factory=list)
    iter_evals: list[int] = field(default_factory=list)
    sigma_trace: list[float] = field(default_factory=list)
    mean_norm_trace: list[float] = field(default_factory=list)
    c_trace: list[float] = field(default_factory=list)

    # Bit-equivalence diff (filled by the runner pairing framework & reference).
    max_mean_diff: float = 0.0
    max_C_diff: float = 0.0
    max_pc_diff: float = 0.0
    max_psigma_diff: float = 0.0
    max_sigma_diff: float = 0.0


# ---------------------------------------------------------------------------
# Reference-driver mirror of the framework's ``optimize`` loop.
# ---------------------------------------------------------------------------


def _run_reference(
    func: BenchmarkFunction,
    initial_point: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    sigma: float,
    population_size: int,
    budget: int,
    seed: int,
) -> tuple[RunRecord, list[dict]]:
    """Run the reference ``CMA`` with the same per-generation accounting
    as :meth:`CMAESOptimizer.optimize` so that comparisons are fair.

    Returns the run record and a per-generation snapshot list that the
    framework run can be diffed against.
    """
    record = RunRecord(
        name="reference",
        fn_name=func.__class__.__name__,
        dim=len(initial_point),
        seed=seed,
    )

    bounds = np.column_stack((lower, upper))
    rng = np.random.default_rng(seed)
    ref = CMA(
        mean=initial_point.copy(),
        sigma=sigma,
        bounds=bounds,
        population_size=population_size,
        seed=rng,
    )

    snapshots: list[dict] = []
    evals = 0
    best_fitness = math.inf

    start = time.perf_counter()
    while evals < budget:
        solutions: list[tuple[NDArray[np.float64], float]] = []
        for _ in range(ref.population_size):
            x = ref.ask()
            f = func(x)
            evals += 1
            solutions.append((x, f))
            best_fitness = min(best_fitness, f)

        # Mean evaluation matches what the framework optimizer does for logging
        # so both sides consume the budget at the same rate.
        mean_repaired = np.clip(ref._mean, lower, upper)
        _ = func(mean_repaired)
        evals += 1

        ref.tell(solutions)

        record.iter_best.append(best_fitness)
        record.iter_evals.append(evals)
        record.sigma_trace.append(float(ref._sigma))
        record.mean_norm_trace.append(float(np.linalg.norm(ref._mean)))
        record.c_trace.append(float(np.trace(ref._C)))

        snapshots.append(
            {
                "sigma": float(ref._sigma),
                "mean": ref._mean.copy(),
                "C": ref._C.copy(),
                "pc": ref._pc.copy(),
                "p_sigma": ref._p_sigma.copy(),
            }
        )

        if ref.should_stop():
            record.converged_message = "internal termination"
            break

    record.wall_time = time.perf_counter() - start
    record.best_fitness = best_fitness
    record.evaluations = evals
    return record, snapshots


def _sample_reference_matched(
    opt: CMAESOptimizer,
    n_max_resampling: int = 100,
) -> NDArray[np.float64]:
    """Sample ``λ`` candidates the way ``CMA.ask`` does it.

    Each individual is drawn with its own ``standard_normal(dim)`` call
    (matching the reference's per-individual RNG sequence) and resampled
    up to ``n_max_resampling`` times before falling back to clamp repair
    (matching the reference's reject-then-clamp semantics).  Reaches into
    private optimizer state because cross-validation needs to mirror the
    reference's sampling loop exactly.
    """
    n = opt.dimensions
    lambda_ = opt.config.population_size
    B, D = opt._eigen_decomposition()  # type: ignore[attr-defined]
    handler = opt.constraint_handler

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        BD = np.dot(B, np.diag(D))

    population = np.empty((lambda_, n))
    for k in range(lambda_):
        chosen: NDArray[np.float64] | None = None
        for _ in range(n_max_resampling):
            z = opt.rng.standard_normal(n)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                y = BD.dot(z)
            candidate = opt._mean + opt._sigma * y  # type: ignore[attr-defined]
            if handler.is_feasible(candidate):
                chosen = candidate
                break
        if chosen is None:
            z = opt.rng.standard_normal(n)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                y = BD.dot(z)
            candidate = opt._mean + opt._sigma * y  # type: ignore[attr-defined]
            # Mirror CMA._repair_infeasible_params (np.where clamp; no inf/nan handling).
            candidate = np.where(
                candidate < handler.lower_bounds, handler.lower_bounds, candidate
            )
            candidate = np.where(
                candidate > handler.upper_bounds, handler.upper_bounds, candidate
            )
            chosen = candidate
        population[k] = chosen
    return population


def _run_framework(
    func: BenchmarkFunction,
    initial_point: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    sigma: float,
    population_size: int,
    budget: int,
    seed: int,
) -> tuple[RunRecord, list[dict]]:
    """Run the framework-native ``CMAESOptimizer`` and harvest snapshots
    after each ``tell`` for direct diffing against the reference."""
    record = RunRecord(
        name="framework",
        fn_name=func.__class__.__name__,
        dim=len(initial_point),
        seed=seed,
    )

    cfg = CMAESConfig(dimensions=len(initial_point))
    cfg.sigma = sigma
    cfg.population_size = population_size

    rng = np.random.default_rng(seed)
    handler = BoxConstraintHandler(BoxStrategy.CLAMP, lower, upper)
    opt: CMAESOptimizer = CMAESOptimizer(
        func=func,
        initial_point=initial_point.copy(),
        config=cfg,
        stopping_condition=MaxEvaluations(budget),
        constraint_handler=handler,
        lower_bounds=lower,
        upper_bounds=upper,
        seed=rng,
    )

    snapshots: list[dict] = []

    # Drive the algorithm by hand with reference-matched sampling so the
    # RNG sequence and repair semantics line up with CMA.ask.  See the
    # module docstring for why this differs from the framework's default
    # sampling path.
    best_fitness = math.inf
    start = time.perf_counter()
    while opt.evaluations < budget:
        pop = _sample_reference_matched(opt)

        fit = np.empty(population_size)
        for k in range(population_size):
            fit[k] = opt.evaluate(pop[k])
            if fit[k] < best_fitness:
                best_fitness = float(fit[k])

        # Mean-fitness diagnostic eval (matches optimize()).
        _ = opt.evaluate(opt.constraint_handler.repair(opt._mean))  # type: ignore[attr-defined]

        opt._tell(pop, fit)  # type: ignore[attr-defined]

        record.iter_best.append(best_fitness)
        record.iter_evals.append(opt.evaluations)
        record.sigma_trace.append(float(opt._sigma))  # type: ignore[attr-defined]
        record.mean_norm_trace.append(float(np.linalg.norm(opt._mean)))  # type: ignore[attr-defined]
        record.c_trace.append(float(np.trace(opt._C)))  # type: ignore[attr-defined]

        snapshots.append(
            {
                "sigma": float(opt._sigma),  # type: ignore[attr-defined]
                "mean": opt._mean.copy(),  # type: ignore[attr-defined]
                "C": opt._C.copy(),  # type: ignore[attr-defined]
                "pc": opt._pc.copy(),  # type: ignore[attr-defined]
                "p_sigma": opt._p_sigma.copy(),  # type: ignore[attr-defined]
            }
        )

        reason = opt._termination_reason()  # type: ignore[attr-defined]
        if reason is not None:
            record.converged_message = reason
            break

    record.wall_time = time.perf_counter() - start
    record.best_fitness = best_fitness
    record.evaluations = opt.evaluations
    return record, snapshots


def _diff_snapshots(
    fw_snapshots: list[dict], ref_snapshots: list[dict]
) -> dict[str, float]:
    """Return the maximum absolute element-wise difference across all
    paired snapshots for sigma, mean, C, pc, p_sigma."""
    pairs = list(zip(fw_snapshots, ref_snapshots))
    if not pairs:
        return {"sigma": 0.0, "mean": 0.0, "C": 0.0, "pc": 0.0, "p_sigma": 0.0}

    def mx(key: str) -> float:
        return max(
            float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))))
            for a, b in pairs
        )

    return {
        "sigma": mx("sigma"),
        "mean": mx("mean"),
        "C": mx("C"),
        "pc": mx("pc"),
        "p_sigma": mx("p_sigma"),
    }


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def _plot_convergence(
    fn_name: str,
    dim: int,
    fw_runs: list[RunRecord],
    ref_runs: list[RunRecord],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in fw_runs:
        ax.plot(
            r.iter_evals, np.maximum(r.iter_best, 1e-300), color="C0", alpha=0.4, lw=1
        )
    for r in ref_runs:
        ax.plot(
            r.iter_evals,
            np.maximum(r.iter_best, 1e-300),
            color="C3",
            alpha=0.4,
            lw=1,
            linestyle="--",
        )

    ax.plot([], [], color="C0", lw=2, label="framework-native")
    ax.plot([], [], color="C3", lw=2, linestyle="--", label="reference port")

    ax.set_xlabel("function evaluations")
    ax.set_ylabel("best fitness so far")
    ax.set_yscale("log")
    ax.set_title(f"{fn_name} — d={dim} (lower is better)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_state_trajectories(
    fn_name: str,
    dim: int,
    fw: RunRecord,
    ref: RunRecord,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, key, label in zip(
        axes,
        ("sigma_trace", "mean_norm_trace", "c_trace"),
        ("σ (step size)", "‖mean‖₂", "tr(C)"),
    ):
        ax.plot(getattr(fw, key), label="framework", color="C0", lw=1.4)
        ax.plot(
            getattr(ref, key), label="reference", color="C3", lw=1.2, linestyle="--"
        )
        ax.set_yscale("log")
        ax.set_xlabel("generation")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"State trajectories — {fn_name}, d={dim}, seed={fw.seed}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_diff_heatmap(rows: list[dict], out_path: Path) -> None:
    """Heatmap of log10(max-diff) across all configurations."""
    df = pd.DataFrame(rows)
    if df.empty:
        return
    # Aggregate over seeds: max across seeds for each (function × dim).
    grouped = df.groupby(["function", "dim"]).agg(
        {
            "max_C_diff": "max",
            "max_mean_diff": "max",
            "max_sigma_diff": "max",
            "max_pc_diff": "max",
            "max_psigma_diff": "max",
        }
    )

    matrix = np.log10(np.maximum(grouped.values.astype(float), 1e-300))  # type: ignore[arg-type]
    labels = [f"{f}\n d={d}" for f, d in grouped.index]  # type: ignore[misc]
    cols = ["C", "mean", "σ", "pc", "p_σ"]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * len(labels))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.0f}",
                ha="center",
                va="center",
                color="w" if matrix[i, j] < matrix.max() - 1 else "k",
                fontsize=8,
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(max |framework − reference|)")
    ax.set_title("Cross-validation equivalence (per-element max diff over the run)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


@dataclass
class FunctionSpec:
    # ``cls`` may be a class (instantiated as ``cls(dim)``) or a zero-arg
    # callable (e.g. a lambda that captures a configured CEC17 function).
    cls: Callable[..., BenchmarkFunction]
    dim: int
    initial: NDArray[np.float64]
    sigma: float
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    budget_factor: int = 1000  # evals/dim

    @property
    def budget(self) -> int:
        return self.budget_factor * self.dim


def build_specs() -> list[FunctionSpec]:
    """A small but representative battery — convex, ill-conditioned,
    non-convex, highly multimodal, and one CEC2017 case."""
    specs: list[FunctionSpec] = []
    for cls, dim, init_val, sigma, bound in (
        (Sphere, 10, 1.0, 0.5, 5.0),
        (Ellipsoid, 10, 1.0, 0.5, 5.0),
        (Rosenbrock, 10, 0.5, 0.5, 5.0),
        (Rastrigin, 10, 1.5, 0.5, 5.12),
        (Ackley, 10, 1.5, 0.5, 32.0),
        (Sphere, 30, 1.0, 0.5, 5.0),
    ):
        init = np.full(dim, init_val)
        lower = np.full(dim, -bound)
        upper = np.full(dim, bound)
        specs.append(
            FunctionSpec(
                cls=cls, dim=dim, initial=init, sigma=sigma, lower=lower, upper=upper
            )
        )
    # One CEC2017 problem.
    dim = 10
    specs.append(
        FunctionSpec(
            cls=lambda d=dim: CEC17Function(dimensions=d, function_id=10),
            dim=dim,
            initial=np.full(dim, 50.0),
            sigma=20.0,
            lower=np.full(dim, -100.0),
            upper=np.full(dim, 100.0),
            budget_factor=500,
        )
    )
    return specs


def _instantiate(spec: FunctionSpec) -> tuple[str, BenchmarkFunction]:
    fn = (
        spec.cls()
        if callable(spec.cls) and not isinstance(spec.cls, type)
        else spec.cls(spec.dim)
    )
    if isinstance(spec.cls, type):
        name = spec.cls.__name__
    else:
        name = fn.__class__.__name__
        if isinstance(fn, CEC17Function):
            name = f"CEC17_F{fn.function_id}"
    return name, fn


def run(seeds: list[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    population_size_for_dim = lambda d: 4 + math.floor(3 * math.log(d))

    for spec in build_specs():
        name, _ = _instantiate(spec)
        fw_runs: list[RunRecord] = []
        ref_runs: list[RunRecord] = []
        diff_first_seed: dict[str, float] = {}

        for seed in seeds:
            fn = (
                spec.cls()
                if callable(spec.cls) and not isinstance(spec.cls, type)
                else spec.cls(spec.dim)
            )
            pop_size = population_size_for_dim(spec.dim)

            fw, fw_snaps = _run_framework(
                func=fn,
                initial_point=spec.initial,
                lower=spec.lower,
                upper=spec.upper,
                sigma=spec.sigma,
                population_size=pop_size,
                budget=spec.budget,
                seed=seed,
            )
            fw.fn_name = name

            ref, ref_snaps = _run_reference(
                func=fn,
                initial_point=spec.initial,
                lower=spec.lower,
                upper=spec.upper,
                sigma=spec.sigma,
                population_size=pop_size,
                budget=spec.budget,
                seed=seed,
            )
            ref.fn_name = name

            diffs = _diff_snapshots(fw_snaps, ref_snaps)
            fw.max_sigma_diff = diffs["sigma"]
            fw.max_mean_diff = diffs["mean"]
            fw.max_C_diff = diffs["C"]
            fw.max_pc_diff = diffs["pc"]
            fw.max_psigma_diff = diffs["p_sigma"]

            fw_runs.append(fw)
            ref_runs.append(ref)
            if not diff_first_seed:
                diff_first_seed = diffs

            summary_rows.append(
                {
                    "function": name,
                    "dim": spec.dim,
                    "seed": seed,
                    "fw_best_fitness": fw.best_fitness,
                    "ref_best_fitness": ref.best_fitness,
                    "abs_best_diff": abs(fw.best_fitness - ref.best_fitness),
                    "fw_evaluations": fw.evaluations,
                    "ref_evaluations": ref.evaluations,
                    "fw_wall_time_s": fw.wall_time,
                    "ref_wall_time_s": ref.wall_time,
                    "max_sigma_diff": diffs["sigma"],
                    "max_mean_diff": diffs["mean"],
                    "max_C_diff": diffs["C"],
                    "max_pc_diff": diffs["pc"],
                    "max_psigma_diff": diffs["p_sigma"],
                }
            )

        # Per-function convergence overlay
        _plot_convergence(
            fn_name=name,
            dim=spec.dim,
            fw_runs=fw_runs,
            ref_runs=ref_runs,
            out_path=output_dir / f"convergence_{name}_d{spec.dim}.png",
        )

        # State trajectories — single seed (first), since equivalent runs
        # produce identical traces and the visual diff is uninformative
        # otherwise.
        _plot_state_trajectories(
            fn_name=name,
            dim=spec.dim,
            fw=fw_runs[0],
            ref=ref_runs[0],
            out_path=output_dir / f"state_{name}_d{spec.dim}.png",
        )

        print(
            f"  {name:<14} d={spec.dim:<3}  "
            f"max|Δσ|={diff_first_seed['sigma']:.2e}  "
            f"max|Δmean|={diff_first_seed['mean']:.2e}  "
            f"max|ΔC|={diff_first_seed['C']:.2e}  "
            f"fw_best={fw_runs[0].best_fitness:.3e}  "
            f"ref_best={ref_runs[0].best_fitness:.3e}"
        )

    df = pd.DataFrame(summary_rows)
    df.to_csv(output_dir / "summary.csv", index=False)
    _plot_diff_heatmap(summary_rows, output_dir / "equivalence_heatmap.png")

    print("\n=== Summary ===")
    grouped = df.groupby(["function", "dim"])[
        [
            "abs_best_diff",
            "max_sigma_diff",
            "max_mean_diff",
            "max_C_diff",
            "max_pc_diff",
            "max_psigma_diff",
        ]
    ].max()
    pd.set_option("display.float_format", lambda x: f"{x:.3e}")
    print(grouped.to_string())
    print(f"\nOutputs written to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds to run")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/cross_validation/cmaes_vs_reference"),
    )
    args = parser.parse_args()
    seeds = list(range(1, args.seeds + 1))
    run(seeds=seeds, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
