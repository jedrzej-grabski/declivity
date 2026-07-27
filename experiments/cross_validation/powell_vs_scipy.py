from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.powell.config import PowellConfig
from declivity.core.algorithm_factory import AlgorithmFactory
from declivity.plotting import plot_metrics
from declivity.plotting.types import PanelSet
from declivity.utils.benchmark_functions import (
    BenchmarkFunction,
    Ellipsoid,
    Rosenbrock,
    Sphere,
)
from declivity.utils.stopping_conditions import MaxEvaluations

PLOTS_DIR = Path("plots/cross_validation/powell_vs_scipy")

XTOL = 1e-4
FTOL = 1e-4


# ---------------------------------------------------------------------------
# SciPy runner with frame-introspection state probe.
# ---------------------------------------------------------------------------


@dataclass
class ScipyPowellTrace:
    """Internal state of scipy's ``_minimize_powell``, one record per
    iteration, read out of the live solver frame by the callback."""

    evaluations: list[int] = field(default_factory=list)
    fval: list[float] = field(default_factory=list)
    x: list[NDArray[np.float64]] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    bigind: list[int] = field(default_factory=list)
    step_norm: list[float] = field(default_factory=list)
    direc: list[NDArray[np.float64]] = field(default_factory=list)


def run_scipy_powell(
    func: BenchmarkFunction,
    x0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    budget: int,
):
    """Run unmodified scipy Powell, capturing internal state per iteration."""
    trace = ScipyPowellTrace()

    def callback(intermediate_result):
        # Walk up to the _minimize_powell frame; its locals hold the
        # solver state the public API never exposes.
        frame = inspect.currentframe()
        while frame is not None and "direc" not in frame.f_locals:
            frame = frame.f_back
        if frame is None:
            return
        state = frame.f_locals
        trace.evaluations.append(int(state["fcalls"][0]))
        trace.fval.append(float(state["fval"]))
        trace.x.append(np.array(state["x"], dtype=float))
        trace.delta.append(float(state["delta"]))
        trace.bigind.append(int(state["bigind"]))
        # x1 still holds the sweep's start point at callback time.
        trace.step_norm.append(float(np.linalg.norm(state["x"] - state["x1"])))
        trace.direc.append(np.array(state["direc"], dtype=float))

    result = minimize(
        func,
        np.array(x0, dtype=float),
        method="Powell",
        bounds=Bounds(lower, upper),
        options={
            "xtol": XTOL,
            "ftol": FTOL,
            "maxfev": budget,
            "maxiter": 10**9,
        },
        callback=callback,
    )
    return trace, result


# ---------------------------------------------------------------------------
# Framework runner.
# ---------------------------------------------------------------------------


def run_declivity_powell(
    func: BenchmarkFunction,
    x0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    budget: int,
):
    config = PowellConfig(dimensions=len(x0), xtol=XTOL, ftol=FTOL)
    config.enable_all_diagnostics()
    optimizer = AlgorithmFactory.create_optimizer(
        AlgorithmChoice.POWELL,
        func,
        np.array(x0, dtype=float),
        config=config,
        stopping_condition=MaxEvaluations(budget),
        lower_bounds=lower,
        upper_bounds=upper,
    )
    result = optimizer.optimize()
    return result.diagnostic, result


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------

_OURS_STYLE = dict(color="tab:blue", lw=1.8, label="declivity")
_SCIPY_STYLE = dict(color="tab:orange", lw=1.8, ls="--", label="scipy")


def plot_convergence(logs, trace: ScipyPowellTrace, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(logs.evaluations, np.maximum(logs.best_fitness, 1e-30), **_OURS_STYLE)
    ax.semilogy(trace.evaluations, np.maximum(trace.fval, 1e-30), **_SCIPY_STYLE)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best fitness (log)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_state_trajectories(
    logs, trace: ScipyPowellTrace, title: str, path: Path
) -> None:
    """Six internal-state metrics, declivity vs scipy, per iteration."""
    iters_ours = np.asarray(logs.iteration)
    iters_scipy = np.arange(1, len(trace.fval) + 1)

    scipy_cond = [float(np.linalg.cond(d)) for d in trace.direc]
    scipy_det = [float(abs(np.linalg.det(d))) for d in trace.direc]

    scipy_evals_per_iter = np.diff([1] + trace.evaluations)
    ours_evals_per_iter = np.diff([1] + list(logs.evaluations))

    panels = [
        ("Function value f(x)", "log", (logs.function_value, trace.fval)),
        ("Sweep displacement ||dx||", "log", (logs.step_norm, trace.step_norm)),
        ("Largest single-direction decrease", "log", (logs.delta, trace.delta)),
        (
            "Direction-set condition number",
            "log",
            (logs.direc_condition_number, scipy_cond),
        ),
        ("Direction-set |det|", "log", (logs.direc_determinant, scipy_det)),
        (
            "Evaluations per iteration",
            "linear",
            (ours_evals_per_iter, scipy_evals_per_iter),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (panel_title, yscale, (ours, scipys)) in zip(axes.flat, panels):
        ours = np.asarray(ours, dtype=float)
        scipys = np.asarray(scipys, dtype=float)
        if yscale == "log":
            ours = np.maximum(ours, 1e-30)
            scipys = np.maximum(scipys, 1e-30)
        ax.plot(iters_ours[: len(ours)], ours, **_OURS_STYLE)
        ax.plot(iters_scipy[: len(scipys)], scipys, **_SCIPY_STYLE)
        ax.set_yscale(yscale)
        ax.set_title(panel_title)
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trace_deviation_series(
    a: ScipyPowellTrace, b: ScipyPowellTrace
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Per-iteration deviation between two SciPy traces (used for the
    chaos sensitivity reference: scipy vs scipy with perturbed x0)."""
    n = min(len(a.fval), len(b.fval))
    f_diff = np.abs(np.asarray(a.fval[:n]) - np.asarray(b.fval[:n])) / np.maximum(
        np.abs(np.asarray(b.fval[:n])), 1e-12
    )
    x_diff = np.linalg.norm(np.asarray(a.x[:n]) - np.asarray(b.x[:n]), axis=1)
    direc_diff = np.array(
        [float(np.max(np.abs(p - q))) for p, q in zip(a.direc[:n], b.direc[:n])]
    )
    evals_diff = np.abs(
        np.asarray(a.evaluations[:n], dtype=float)
        - np.asarray(b.evaluations[:n], dtype=float)
    )
    return f_diff, x_diff, direc_diff, evals_diff


def plot_divergence(
    logs,
    trace: ScipyPowellTrace,
    chaos_reference: tuple[NDArray, NDArray, NDArray, NDArray] | None,
    title: str,
    path: Path,
) -> dict[str, float]:

    n = min(len(logs.iteration), len(trace.fval))
    floor = 1e-18

    f_ours = np.asarray(logs.function_value[:n])
    f_scipy = np.asarray(trace.fval[:n])
    f_diff = np.abs(f_ours - f_scipy) / np.maximum(np.abs(f_scipy), 1e-12)

    x_ours = np.asarray(logs.current_point[:n])
    x_scipy = np.asarray(trace.x[:n])
    x_diff = np.linalg.norm(x_ours - x_scipy, axis=1)

    direc_ours = logs.direction_set[:n]
    direc_scipy = trace.direc[:n]
    direc_diff = np.array(
        [float(np.max(np.abs(a - b))) for a, b in zip(direc_ours, direc_scipy)]
    )

    evals_diff = np.abs(
        np.asarray(logs.evaluations[:n]) - np.asarray(trace.evaluations[:n])
    )

    iters = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    exact = (
        np.max(f_diff, initial=0) == 0
        and np.max(x_diff, initial=0) == 0
        and np.max(direc_diff, initial=0) == 0
        and np.max(evals_diff, initial=0) == 0
    )
    ours_label_suffix = "  (≡ 0, exact)" if exact else ""
    ax.semilogy(
        iters,
        np.maximum(f_diff, floor),
        label=f"declivity vs scipy: |f diff| (rel){ours_label_suffix}",
        color="tab:blue",
        lw=2.0,
    )
    ax.semilogy(
        iters,
        np.maximum(direc_diff, floor),
        label=f"declivity vs scipy: max |direc diff|{ours_label_suffix}",
        color="tab:cyan",
        lw=2.0,
        ls="-.",
    )

    if chaos_reference is not None:
        cf, cx, cdirec, _ = chaos_reference
        m = len(cf)
        c_iters = np.arange(1, m + 1)
        ax.semilogy(
            c_iters,
            np.maximum(cf, floor),
            label="chaos ref — scipy vs scipy(x0+1e-12): |f diff| (rel)",
            color="tab:red",
            lw=1.4,
            ls="--",
            alpha=0.8,
        )
        ax.semilogy(
            c_iters,
            np.maximum(cdirec, floor),
            label="chaos ref — scipy vs scipy(x0+1e-12): max |direc diff|",
            color="tab:orange",
            lw=1.4,
            ls=":",
            alpha=0.8,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Deviation (log; floored at 1e-18)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "max_f_rel_diff": float(np.max(f_diff)) if n else float("nan"),
        "max_x_diff": float(np.max(x_diff)) if n else float("nan"),
        "max_direc_diff": float(np.max(direc_diff)) if n else float("nan"),
        "max_evals_diff": int(np.max(evals_diff)) if n else -1,
    }


def plot_agreement_matrix(summary: pd.DataFrame, path: Path, title: str) -> None:
    """One glanceable figure: max deviation across all seeds and iterations
    per (function x dim) row and metric column.  All-zero cells are the
    headline result — annotated '0 (exact)'."""
    metrics = ["max_f_rel_diff", "max_x_diff", "max_direc_diff", "max_evals_diff"]
    labels = ["|f diff| (rel)", "||x diff||", "max |direc diff|", "|evals diff|"]

    grouped = summary.groupby(["function", "dim"])[metrics].max()
    values = grouped.to_numpy(dtype=float)
    log_values = np.log10(np.maximum(np.abs(values), 1e-18))

    fig, ax = plt.subplots(
        figsize=(2.1 * len(metrics) + 3.5, 0.65 * len(grouped) + 2.2)
    )
    im = ax.imshow(log_values, cmap="RdYlGn_r", vmin=-18, vmax=0, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(
        range(len(grouped)),
        [f"{fn} d={dim}" for fn, dim in grouped.index],
        fontsize=9,
    )
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = "0 (exact)" if values[i, j] == 0 else f"{values[i, j]:.1e}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    ax.set_title(
        f"{title}\nmax deviation over all seeds and all iterations "
        f"(green = agreement at the 1e-18 floor)"
    )
    fig.colorbar(im, ax=ax, label="log10(max deviation)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--num-seeds", type=int, default=5)
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    functions: list[type[BenchmarkFunction]] = [Sphere, Rosenbrock, Ellipsoid]

    rows = []
    for dim in args.dims:
        budget = 10_000 * dim
        for fn_cls in functions:
            fn = fn_cls(dim)
            lower, upper = fn.bounds
            fn_name = fn_cls.__name__.lower()

            for seed in range(args.num_seeds):
                rng = np.random.default_rng(1000 + seed)
                x0 = rng.uniform(-3.0, 3.0, dim)

                scipy_trace, scipy_result = run_scipy_powell(
                    fn, x0, lower, upper, budget
                )
                logs, our_result = run_declivity_powell(fn, x0, lower, upper, budget)

                tag = f"{fn_name}_d{dim}"
                if seed == 0:
                    plot_convergence(
                        logs,
                        scipy_trace,
                        f"Powell — {fn_cls.__name__} d={dim} (seed 0)",
                        PLOTS_DIR / f"convergence_{tag}.png",
                    )
                    plot_state_trajectories(
                        logs,
                        scipy_trace,
                        f"Powell internal state — {fn_cls.__name__} d={dim} (seed 0)",
                        PLOTS_DIR / f"state_{tag}.png",
                    )
                    plot_metrics(
                        our_result,
                        panels=PanelSet.ALL,
                        title=f"declivity Powell — {fn_cls.__name__} d={dim}",
                        save_path=PLOTS_DIR / f"panels_{tag}.png",
                    )
                    plt.close("all")

                if seed == 0:
                    # Chaos sensitivity reference: the same probe applied
                    # to scipy vs scipy started from x0 + 1e-12.
                    scipy_trace_pert, _ = run_scipy_powell(
                        fn, x0 + 1e-12, lower, upper, budget
                    )
                    chaos_reference = trace_deviation_series(
                        scipy_trace_pert, scipy_trace
                    )
                    diffs = plot_divergence(
                        logs,
                        scipy_trace,
                        chaos_reference,
                        f"Powell deviation — {fn_cls.__name__} d={dim} (seed {seed})",
                        PLOTS_DIR / f"divergence_{tag}_s{seed}.png",
                    )
                else:
                    diffs = _diffs_only(logs, scipy_trace)

                rows.append(
                    {
                        "function": fn_cls.__name__,
                        "dim": dim,
                        "seed": seed,
                        "f_declivity": our_result.best_fitness,
                        "f_scipy": float(scipy_result.fun),
                        "evals_declivity": our_result.evaluations,
                        "evals_scipy": int(scipy_result.nfev),
                        "iters_declivity": len(logs.iteration),
                        "iters_scipy": len(scipy_trace.fval),
                        **diffs,
                    }
                )
                print(
                    f"[{fn_cls.__name__:10s} d={dim:2d} seed={seed}] "
                    f"f: {our_result.best_fitness:.3e} vs {scipy_result.fun:.3e} | "
                    f"evals: {our_result.evaluations} vs {scipy_result.nfev} | "
                    f"max direc diff: {diffs['max_direc_diff']:.2e}"
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(PLOTS_DIR / "summary.csv", index=False)
    plot_agreement_matrix(
        summary,
        PLOTS_DIR / "agreement_matrix.png",
        "Powell: declivity vs scipy agreement",
    )
    print(f"\nSummary written to {PLOTS_DIR / 'summary.csv'}")
    print(
        summary.groupby(["function", "dim"])[
            ["max_f_rel_diff", "max_x_diff", "max_direc_diff", "max_evals_diff"]
        ].max()
    )


def _diffs_only(logs, trace: ScipyPowellTrace) -> dict[str, float]:
    """Deviation metrics without the plot (non-zero seeds)."""
    n = min(len(logs.iteration), len(trace.fval))
    if n == 0:
        return {
            "max_f_rel_diff": float("nan"),
            "max_x_diff": float("nan"),
            "max_direc_diff": float("nan"),
            "max_evals_diff": -1,
        }
    f_diff = np.abs(
        np.asarray(logs.function_value[:n]) - np.asarray(trace.fval[:n])
    ) / np.maximum(np.abs(np.asarray(trace.fval[:n])), 1e-12)
    x_diff = np.linalg.norm(
        np.asarray(logs.current_point[:n]) - np.asarray(trace.x[:n]), axis=1
    )
    direc_diff = np.array(
        [
            float(np.max(np.abs(a - b)))
            for a, b in zip(logs.direction_set[:n], trace.direc[:n])
        ]
    )
    evals_diff = np.abs(
        np.asarray(logs.evaluations[:n]) - np.asarray(trace.evaluations[:n])
    )
    return {
        "max_f_rel_diff": float(np.max(f_diff)),
        "max_x_diff": float(np.max(x_diff)),
        "max_direc_diff": float(np.max(direc_diff)),
        "max_evals_diff": int(np.max(evals_diff)),
    }


if __name__ == "__main__":
    main()
