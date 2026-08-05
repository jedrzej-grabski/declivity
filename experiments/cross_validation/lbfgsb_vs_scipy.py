"""Cross-validation: framework-native L-BFGS-B vs SciPy's L-BFGS-B.

The SciPy reference is :func:`scipy.optimize.minimize` with
``method="L-BFGS-B"`` — a Python wrapper over the Fortran v3.0
implementation by Zhu, Byrd, Lu, and Nocedal.  This module pits it
against :class:`~declivity.algorithms.lbfgsb.LBFGSBOptimizer`, the pure-Python
reimplementation in declivity.

Because L-BFGS-B is **deterministic** given an initial point, the
notion of "seeds" used by the DES / MF-CMA-ES harnesses does not carry
across cleanly.  Here a *seed* selects an initial point ``x0`` drawn
uniformly from ``[lb, ub]^d`` — both implementations receive that same
``x0`` for the run.  The parity claim is that, across many random
``x0`` draws, the two implementations converge to indistinguishable
distributions of final fitness.  Multimodal problems (Rastrigin,
Ackley) additionally let us check that both impls get trapped at the
*same* local minima, not merely that they hit the optimum on the easy
seeds.

Output (under ``plots/cross_validation/lbfgsb_vs_scipy/``):

- ``convergence_<problem>.png`` — per-seed best-fitness overlay
- ``distribution_<problem>.png`` — final-fitness boxplot with Wilcoxon/KS
- ``summary.csv`` — per-seed final, nfev, niter, ‖x_fw − x_ref‖₂
- ``parity_report.txt`` — descriptive stats + tests

Run::

    PYTHONPATH=. uv run python -m experiments.cross_validation.lbfgsb_vs_scipy --function rosenbrock_d10
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize as scipy_minimize

from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.algorithms.lbfgsb.lbfgsb_optimizer import LBFGSBOptimizer
from declivity.utils.constraint_handlers import BoxConstraintHandler, BoxStrategy
from declivity.utils.gradient_strategies import ForwardFD
from declivity.utils.stopping_conditions import MaxEvaluations
from experiments.cross_validation._problems import PROBLEMS, ProblemSpec

DEFAULT_PROBLEM = "ellipsoid_d10"
DEFAULT_SEEDS = 25
DEFAULT_OUTPUT_DIR = Path("plots/cross_validation/lbfgsb_vs_scipy")


@dataclass
class RunRecord:
    name: str
    seed: int
    x0: NDArray[np.float64]
    best_fit: float = math.inf
    best_x: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    nfev: int = 0
    niter: int = 0
    wall_time: float = 0.0
    iter_best: list[float] = field(default_factory=list)
    iter_evals: list[int] = field(default_factory=list)


def _draw_x0(spec: ProblemSpec, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(spec.lower, spec.upper, size=spec.dim)


def _run_framework(spec: ProblemSpec, seed: int, x0: NDArray[np.float64]) -> RunRecord:
    func = spec.fn_factory(spec.dim)
    cfg = LBFGSBConfig(dimensions=spec.dim)
    cfg.enable_all_diagnostics()

    # Use forward-FD strategy to match scipy's default gradient method;
    # the framework's default (central FD) gives more accurate gradients
    # but uses 2× the evaluations, which would skew the per-evaluation
    # comparison.  The framework default is left unchanged elsewhere.
    opt = LBFGSBOptimizer(
        func=func,
        initial_point=x0,
        config=cfg,
        stopping_condition=MaxEvaluations(10_000 * spec.dim),
        lower_bounds=spec.lower,
        upper_bounds=spec.upper,
        constraint_handler=BoxConstraintHandler(
            BoxStrategy.CLAMP,
            np.full(spec.dim, spec.lower),
            np.full(spec.dim, spec.upper),
        ),
        gradient_strategy=ForwardFD(),
    )

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = opt.optimize()
    elapsed = time.perf_counter() - start

    diag = result.diagnostic
    iter_best = list(diag.best_fitness)
    iter_evals = list(getattr(diag, "evaluations", []) or [])
    if not iter_evals or len(iter_evals) != len(iter_best):
        iter_evals = list(range(1, len(iter_best) + 1))

    return RunRecord(
        name="framework-native",
        seed=seed,
        x0=x0,
        best_fit=result.best_fitness,
        best_x=np.asarray(result.best_solution, dtype=float),
        nfev=result.evaluations,
        niter=len(iter_best),
        wall_time=elapsed,
        iter_best=iter_best,
        iter_evals=iter_evals,
    )


def _run_scipy(spec: ProblemSpec, seed: int, x0: NDArray[np.float64]) -> RunRecord:
    func = spec.fn_factory(spec.dim)

    # Record per-iteration best by snapshotting via callback.  SciPy's
    # L-BFGS-B callback fires once per accepted iteration with the new
    # ``xk`` — we re-evaluate ``func(xk)`` for plot trajectories.  Those
    # evaluations are bookkeeping for the chart only and do not feed
    # back into the algorithm.
    trajectory: list[tuple[int, float]] = []
    eval_counter = [0]

    def wrapped(x: NDArray[np.float64]) -> float:
        eval_counter[0] += 1
        return float(func(x))

    def callback(xk: NDArray[np.float64]) -> None:
        # snapshot the current best-so-far at the iteration boundary
        f_xk = float(func(xk))
        if trajectory and f_xk > trajectory[-1][1]:
            f_xk = trajectory[-1][1]
        trajectory.append((eval_counter[0], f_xk))

    bounds = list(zip(np.full(spec.dim, spec.lower), np.full(spec.dim, spec.upper)))

    start = time.perf_counter()
    result = scipy_minimize(
        fun=wrapped,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={"maxiter": 10_000, "maxfun": 10_000 * spec.dim},
    )
    elapsed = time.perf_counter() - start

    iter_evals = [step[0] for step in trajectory]
    iter_best = [step[1] for step in trajectory]
    if not iter_best:
        iter_evals = [eval_counter[0]]
        iter_best = [float(result.fun)]

    return RunRecord(
        name="scipy",
        seed=seed,
        x0=x0,
        best_fit=float(result.fun),
        best_x=np.asarray(result.x, dtype=float),
        nfev=int(result.nfev),
        niter=int(result.nit),
        wall_time=elapsed,
        iter_best=iter_best,
        iter_evals=iter_evals,
    )


def _plot_convergence(
    spec: ProblemSpec,
    fw_runs: list[RunRecord],
    ref_runs: list[RunRecord],
    out_path: Path,
) -> None:
    """Convergence overlay — per-evaluation x-axis.

    With both implementations using forward FD for the gradient, the
    per-evaluation and per-iteration views are nearly equivalent.  A
    single per-evaluation panel matches the DES / MF-CMA-ES harnesses.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in fw_runs:
        if not r.iter_best:
            continue
        evals = np.asarray(r.iter_evals, dtype=float)
        fit = np.maximum(np.asarray(r.iter_best, dtype=float), spec.floor)
        ax.plot(evals, fit, color="C0", alpha=0.45, lw=0.9)
    for r in ref_runs:
        if not r.iter_best:
            continue
        evals = np.asarray(r.iter_evals, dtype=float)
        fit = np.maximum(np.asarray(r.iter_best, dtype=float), spec.floor)
        ax.plot(evals, fit, color="C3", alpha=0.45, lw=0.9, linestyle="--")

    ax.plot([], [], color="C0", lw=2, label="declivity (framework-native)")
    ax.plot(
        [], [], color="C3", lw=2, linestyle="--", label="scipy.optimize (Fortran v3.0)"
    )
    if spec.f_star > 0:
        ax.axhline(spec.f_star, color="gray", lw=0.8, linestyle=":", alpha=0.6)
        ax.annotate(
            f"f* = {spec.f_star:g}",
            xy=(0.01, spec.f_star),
            xycoords=("axes fraction", "data"),
            color="gray",
            fontsize=8,
            va="bottom",
            ha="left",
        )

    ax.set_xlabel("liczba ewaluacji funkcji celu")
    ax.set_ylabel("najlepsza wartość funkcji celu")
    ax.set_yscale("log")
    ax.set_title(
        f"{spec.name}, d = {spec.dim} — L-BFGS-B: declivity vs scipy "
        f"({len(fw_runs)} losowych punktów startowych)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_distribution(
    spec: ProblemSpec,
    fw_finals: NDArray[np.float64],
    ref_finals: NDArray[np.float64],
    p_w: float,
    p_ks: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(
        [fw_finals, ref_finals],
        tick_labels=["declivity", "scipy"],
        widths=0.4,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], ["C0", "C3"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for median in bp["medians"]:
        median.set_color("black")
    jx = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(fw_finals))
    jy = np.random.default_rng(1).uniform(-0.08, 0.08, size=len(ref_finals))
    ax.scatter(1 + jx, fw_finals, color="C0", s=18, alpha=0.7, zorder=3)
    ax.scatter(2 + jy, ref_finals, color="C3", s=18, alpha=0.7, zorder=3)
    if spec.f_star > 0:
        ax.axhline(spec.f_star, color="gray", lw=0.8, linestyle=":", alpha=0.6)
    finite = np.concatenate(
        [fw_finals[np.isfinite(fw_finals)], ref_finals[np.isfinite(ref_finals)]]
    )
    if (
        finite.size
        and finite.min() > 0
        and finite.max() / max(finite.min(), 1e-300) > 1e3
    ):
        ax.set_yscale("log")
    ax.set_ylabel("najlepsza wartość funkcji celu (koniec uruchomienia)")
    ax.set_title(
        f"Rozkład wyników końcowych — {spec.name}, d = {spec.dim}\n"
        f"Wilcoxon p = {p_w:.3f},   KS p = {p_ks:.3f}"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(seeds: list[int], spec: ProblemSpec, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fw_runs: list[RunRecord] = []
    ref_runs: list[RunRecord] = []

    print(f"# L-BFGS-B parity audit on {spec.name} (d={spec.dim})")
    for seed in seeds:
        x0 = _draw_x0(spec, seed)
        print(f"[seed {seed:2d}] framework…", end=" ", flush=True)
        fw = _run_framework(spec, seed, x0)
        print(
            f"f={fw.best_fit:.3e} (nfev={fw.nfev}, niter={fw.niter}, t={fw.wall_time:.2f}s)",
            end="  ",
            flush=True,
        )
        print("scipy…", end=" ", flush=True)
        ref = _run_scipy(spec, seed, x0)
        print(
            f"f={ref.best_fit:.3e} (nfev={ref.nfev}, niter={ref.niter}, t={ref.wall_time:.2f}s)"
        )
        fw_runs.append(fw)
        ref_runs.append(ref)

    fw_finals = np.array([r.best_fit for r in fw_runs])
    ref_finals = np.array([r.best_fit for r in ref_runs])

    rows = []
    for fw, ref in zip(fw_runs, ref_runs):
        x_diff = float(np.linalg.norm(fw.best_x - ref.best_x))
        rows.append(
            {
                "seed": fw.seed,
                "fw_final": fw.best_fit,
                "ref_final": ref.best_fit,
                "abs_diff_f": abs(fw.best_fit - ref.best_fit),
                "x_diff_l2": x_diff,
                "fw_nfev": fw.nfev,
                "ref_nfev": ref.nfev,
                "fw_niter": fw.niter,
                "ref_niter": ref.niter,
                "fw_wall_s": fw.wall_time,
                "ref_wall_s": ref.wall_time,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "summary.csv", index=False)

    try:
        w_stat, p_w = stats.wilcoxon(fw_finals, ref_finals, zero_method="zsplit")
    except ValueError:
        w_stat, p_w = 0.0, 1.0
    ks_stat, p_ks = stats.ks_2samp(fw_finals, ref_finals)

    median_x_diff = float(np.median([r["x_diff_l2"] for r in rows]))
    max_x_diff = float(np.max([r["x_diff_l2"] for r in rows]))

    report = [
        f"# L-BFGS-B parity report — {spec.name}, d={spec.dim}, n={len(seeds)} seeds",
        "",
        "## Descriptive statistics on final fitness",
        f"declivity:  mean={fw_finals.mean():.4e}  median={np.median(fw_finals):.4e}  "
        f"std={fw_finals.std(ddof=1):.4e}  min={fw_finals.min():.4e}  max={fw_finals.max():.4e}",
        f"scipy:      mean={ref_finals.mean():.4e}  median={np.median(ref_finals):.4e}  "
        f"std={ref_finals.std(ddof=1):.4e}  min={ref_finals.min():.4e}  max={ref_finals.max():.4e}",
        "",
        "## Per-seed L2 distance between final x*",
        f"median ‖x_decl − x_scipy‖₂ = {median_x_diff:.4e}",
        f"max    ‖x_decl − x_scipy‖₂ = {max_x_diff:.4e}",
        "",
        "## Statistical tests on final fitness (H0: same distribution)",
        f"Wilcoxon signed-rank:  W={w_stat:.3f}  p={p_w:.4f}  "
        f"=> {'distributions indistinguishable' if p_w > 0.05 else 'DIFFER (p < 0.05)'}",
        f"Kolmogorov–Smirnov 2s: D={ks_stat:.3f}  p={p_ks:.4f}  "
        f"=> {'distributions indistinguishable' if p_ks > 0.05 else 'DIFFER (p < 0.05)'}",
        "",
        "## Evaluation budget",
        f"declivity median nfev: {int(np.median([r.nfev for r in fw_runs]))}  "
        f"scipy median nfev: {int(np.median([r.nfev for r in ref_runs]))}",
        "",
    ]
    report_text = "\n".join(report)
    (output_dir / "parity_report.txt").write_text(report_text)
    print()
    print(report_text)

    tag = f"{spec.name}_d{spec.dim}"
    _plot_convergence(spec, fw_runs, ref_runs, output_dir / f"convergence_{tag}.png")
    _plot_distribution(
        spec,
        fw_finals,
        ref_finals,
        float(p_w),
        float(p_ks),
        output_dir / f"distribution_{tag}.png",
    )
    print(f"wrote plots to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--function",
        type=str,
        default=DEFAULT_PROBLEM,
        choices=sorted(PROBLEMS.keys()),
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    spec = PROBLEMS[args.function]
    run(seeds=seeds, spec=spec, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
