"""Cross-validation: framework-native DES vs the literal port of DES.R.

Mirrors :mod:`cmaes_vs_reference`.  The "R reference" here is
:func:`src.algorithms.des.des_reference.des_reference` — a line-by-line
Python translation of ``thesis-experiments/des_comparison/DES.R``,
driven with the same NumPy seed as the framework-native
:class:`src.algorithms.des.des_optimizer.DESOptimizer`.

Parity is assessed distributionally over many seeds: convergence
trajectories should overlap, and the final-fitness samples should be
indistinguishable by Wilcoxon signed-rank and two-sample KS tests at
α = 0.05.  RNG-draw order across the two implementations is **not**
required to bit-match (the framework optimizer threads RNG calls through
:class:`PopulationInitializer` and other seams that the literal port
sidesteps), so only the distributional claim is meaningful.

Output (under ``plots/cross_validation/des_vs_reference/``):

- ``convergence_CEC17_F<id>_d<dim>.png`` — per-seed best-fitness overlay
- ``distribution_CEC17_F<id>_d<dim>.png`` — final-fitness box + strip plot
- ``state_CEC17_F<id>_d<dim>.png`` — mean-norm / pc-norm / popMean-norm
  side-by-side for seed 0
- ``summary.csv`` — per-seed final fitness, evaluations, wall time
- ``parity_report.txt`` — Wilcoxon / KS / descriptive stats

Run::

    PYTHONPATH=. uv run python -m experiments.cross_validation.des_vs_reference
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.des.config import DESConfig
from declivity.algorithms.des.des_reference import des_reference
from declivity.core.algorithm_factory import AlgorithmFactory
from declivity.utils.constraint_handlers import BoxConstraintHandler, BoxStrategy
from declivity.utils.stopping_conditions import MaxEvaluations
from experiments.cross_validation._problems import PROBLEMS, ProblemSpec

# Default problem: CEC2017 F10 in 10 dimensions, the existing parity
# anchor in the thesis (§3.6) and the R harness in ``DES.R``.
DEFAULT_PROBLEM = "cec17_F10_d10"
DEFAULT_SEEDS = 25
DEFAULT_OUTPUT_DIR = Path("plots/cross_validation/des_vs_reference")


@dataclass
class RunRecord:
    name: str
    seed: int
    best_fit: float = math.inf
    evaluations: int = 0
    wall_time: float = 0.0
    iter_best: list[float] = field(default_factory=list)
    iter_evals: list[int] = field(default_factory=list)
    ft_history: list[float] = field(default_factory=list)
    new_mean_norm: list[float] = field(default_factory=list)
    pc_norm: list[float] = field(default_factory=list)
    pop_mean_norm: list[float] = field(default_factory=list)


def _run_reference(spec: ProblemSpec, seed: int) -> RunRecord:
    func = spec.fn_factory(spec.dim)
    start = time.perf_counter()
    res = des_reference(
        par=spec.x0_factory(spec.dim),
        fn=func,
        lower=spec.lower,
        upper=spec.upper,
        budget=10_000 * spec.dim,
        lambda_=4 * spec.dim,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    return RunRecord(
        name="reference port",
        seed=seed,
        best_fit=res.best_fit,
        evaluations=res.counteval,
        wall_time=elapsed,
        iter_best=res.iter_best,
        iter_evals=res.iter_evals,
        ft_history=res.ft_history,
        new_mean_norm=res.new_mean_norm,
        pc_norm=res.pc_norm,
        pop_mean_norm=res.pop_mean_norm,
    )


def _run_framework(spec: ProblemSpec, seed: int) -> RunRecord:
    func = spec.fn_factory(spec.dim)
    cfg = DESConfig(dimensions=spec.dim)
    cfg.population_size = 4 * spec.dim
    cfg.enable_all_diagnostics()

    rng = np.random.default_rng(seed)
    opt = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.DES,
        func=func,
        initial_point=spec.x0_factory(spec.dim),
        config=cfg,
        stopping_condition=MaxEvaluations(10_000 * spec.dim),
        lower_bounds=spec.lower,
        upper_bounds=spec.upper,
        constraint_handler=BoxConstraintHandler(
            BoxStrategy.BOUNCE_BACK,
            np.full(spec.dim, spec.lower),
            np.full(spec.dim, spec.upper),
        ),
        seed=rng,
    )

    start = time.perf_counter()
    result = opt.optimize()
    elapsed = time.perf_counter() - start

    diag = result.diagnostic
    iter_best = list(diag.best_fitness)
    iter_evals = list(getattr(diag, "evaluations", []) or [0] * len(iter_best))
    if not iter_evals or all(e == 0 for e in iter_evals):
        # PopulationLogData does not always carry per-iter eval counts;
        # approximate from population_size since each inner iter
        # consumes (pop_size + 1) evaluations (pop + cumMean check).
        per_iter = cfg.population_size + 1
        iter_evals = [(i + 1) * per_iter for i in range(len(iter_best))]
    ft_history = list(getattr(diag, "ft", []) or [])
    return RunRecord(
        name="framework-native",
        seed=seed,
        best_fit=result.best_fitness,
        evaluations=result.evaluations,
        wall_time=elapsed,
        iter_best=iter_best,
        iter_evals=iter_evals,
        ft_history=ft_history,
        new_mean_norm=[],
        pc_norm=[],
        pop_mean_norm=[],
    )


def _plot_convergence(
    spec: ProblemSpec,
    fw_runs: list[RunRecord],
    ref_runs: list[RunRecord],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in fw_runs:
        if not r.iter_best:
            continue
        evals = np.asarray(r.iter_evals, dtype=float)
        fit = np.maximum(np.asarray(r.iter_best, dtype=float), spec.floor)
        ax.plot(evals, fit, color="C0", alpha=0.35, lw=0.9)

    for r in ref_runs:
        if not r.iter_best:
            continue
        evals = np.asarray(r.iter_evals, dtype=float)
        fit = np.maximum(np.asarray(r.iter_best, dtype=float), spec.floor)
        ax.plot(evals, fit, color="C3", alpha=0.35, lw=0.9, linestyle="--")

    ax.plot([], [], color="C0", lw=2, label="declivity (framework-native)")
    ax.plot([], [], color="C3", lw=2, linestyle="--", label="port DES.R (referencja)")
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
        f"{spec.name}, d = {spec.dim} — DES: declivity vs port DES.R "
        f"({len(fw_runs)} ziaren)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_distribution(
    spec: ProblemSpec,
    fw_finals: NDArray[np.float64],
    ref_finals: NDArray[np.float64],
    p_wilcoxon: float,
    p_ks: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    box_data = [fw_finals, ref_finals]
    labels = ["declivity", "port DES.R"]
    bp = ax.boxplot(
        box_data, tick_labels=labels, widths=0.4, patch_artist=True, showfliers=False
    )
    for patch, color in zip(bp["boxes"], ["C0", "C3"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for median in bp["medians"]:
        median.set_color("black")

    jitter_fw = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(fw_finals))
    jitter_rf = np.random.default_rng(1).uniform(-0.08, 0.08, size=len(ref_finals))
    ax.scatter(1 + jitter_fw, fw_finals, color="C0", s=18, alpha=0.7, zorder=3)
    ax.scatter(2 + jitter_rf, ref_finals, color="C3", s=18, alpha=0.7, zorder=3)

    if spec.f_star > 0:
        ax.axhline(spec.f_star, color="gray", lw=0.8, linestyle=":", alpha=0.6)
    # Use log scale when fitness spans several orders of magnitude.
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
        f"Wilcoxon p = {p_wilcoxon:.3f},   KS p = {p_ks:.3f}"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_state(
    spec: ProblemSpec, fw: RunRecord, ref: RunRecord, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    if fw.ft_history:
        ax.plot(fw.ft_history, color="C0", lw=1.2, label="declivity")
    if ref.ft_history:
        ax.plot(ref.ft_history, color="C3", lw=1.0, linestyle="--", label="port")
    ax.set_yscale("log")
    ax.set_xlabel("iteracja")
    ax.set_ylabel("Ft")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    if ref.new_mean_norm:
        ax.plot(ref.new_mean_norm, color="C3", lw=1.0, linestyle="--", label="port")
    ax.set_xlabel("iteracja")
    ax.set_ylabel("‖newMean‖₂")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[2]
    if ref.pc_norm:
        ax.plot(ref.pc_norm, color="C3", lw=1.0, linestyle="--", label="port")
    ax.set_xlabel("iteracja")
    ax.set_ylabel("‖pc‖₂")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Trajektorie stanu wewnętrznego — {spec.name}, d = {spec.dim}, seed = {fw.seed}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(seeds: list[int], spec: ProblemSpec, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fw_runs: list[RunRecord] = []
    ref_runs: list[RunRecord] = []

    print(f"# DES parity audit on {spec.name} (d={spec.dim})")
    for seed in seeds:
        print(f"[seed {seed:2d}] framework…", end=" ", flush=True)
        fw = _run_framework(spec, seed)
        print(
            f"f={fw.best_fit:.3e} (evals={fw.evaluations}, t={fw.wall_time:.1f}s)",
            end="  ",
            flush=True,
        )
        print("reference…", end=" ", flush=True)
        ref = _run_reference(spec, seed)
        print(f"f={ref.best_fit:.3e} (evals={ref.evaluations}, t={ref.wall_time:.1f}s)")
        fw_runs.append(fw)
        ref_runs.append(ref)

    fw_finals = np.array([r.best_fit for r in fw_runs])
    ref_finals = np.array([r.best_fit for r in ref_runs])

    rows: list[dict] = []
    for fw, ref in zip(fw_runs, ref_runs):
        rows.append(
            {
                "seed": fw.seed,
                "fw_final": fw.best_fit,
                "ref_final": ref.best_fit,
                "abs_diff": abs(fw.best_fit - ref.best_fit),
                "fw_evals": fw.evaluations,
                "ref_evals": ref.evaluations,
                "fw_wall_s": fw.wall_time,
                "ref_wall_s": ref.wall_time,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "summary.csv", index=False)

    w_stat, p_wilcoxon = stats.wilcoxon(fw_finals, ref_finals, zero_method="zsplit")
    ks_stat, p_ks = stats.ks_2samp(fw_finals, ref_finals)

    report = [
        f"# DES parity report — {spec.name}, d={spec.dim}, n={len(seeds)} seeds",
        "",
        "## Descriptive statistics",
        f"declivity:  mean={fw_finals.mean():.4f}  median={np.median(fw_finals):.4f}  "
        f"std={fw_finals.std(ddof=1):.4f}  min={fw_finals.min():.4f}  max={fw_finals.max():.4f}",
        f"port DES.R: mean={ref_finals.mean():.4f}  median={np.median(ref_finals):.4f}  "
        f"std={ref_finals.std(ddof=1):.4f}  min={ref_finals.min():.4f}  max={ref_finals.max():.4f}",
        "",
        "## Statistical tests on final fitness (H0: same distribution)",
        f"Wilcoxon signed-rank:  W={w_stat:.3f}  p={p_wilcoxon:.4f}  "
        f"=> {'distributions indistinguishable' if p_wilcoxon > 0.05 else 'DIFFER (p < 0.05)'}",
        f"Kolmogorov–Smirnov 2s: D={ks_stat:.3f}  p={p_ks:.4f}  "
        f"=> {'distributions indistinguishable' if p_ks > 0.05 else 'DIFFER (p < 0.05)'}",
        "",
    ]
    report_text = "\n".join(report)
    (output_dir / "parity_report.txt").write_text(report_text)
    print()
    print(report_text)

    fn_tag = f"{spec.name}_d{spec.dim}"
    _plot_convergence(spec, fw_runs, ref_runs, output_dir / f"convergence_{fn_tag}.png")
    _plot_distribution(
        spec,
        fw_finals,
        ref_finals,
        float(p_wilcoxon),
        float(p_ks),
        output_dir / f"distribution_{fn_tag}.png",
    )
    _plot_state(spec, fw_runs[0], ref_runs[0], output_dir / f"state_{fn_tag}.png")
    print(f"wrote plots to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--function",
        type=str,
        default=DEFAULT_PROBLEM,
        choices=sorted(PROBLEMS.keys()),
        help="benchmark problem from experiments.cross_validation._problems.PROBLEMS",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    spec = PROBLEMS[args.function]
    run(seeds=seeds, spec=spec, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
