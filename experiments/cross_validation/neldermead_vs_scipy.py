"""Cross-validation: framework-native Nelder-Mead vs ``scipy.optimize.minimize(method='Nelder-Mead')``.

The framework :class:`~declivity.algorithms.neldermead.NelderMeadOptimizer`
treats the simplex as a population: the initial simplex comes from the
injected :class:`~declivity.utils.population_initializers.SimplexPopulationInitializer`
(SciPy's deterministic axis-step construction), candidate vertices are
repaired through the injected ``ConstraintHandler`` (CLAMP == SciPy's
bounds clipping), and the shrink step routes through the
``RepairStrategy`` seam.  Both implementations are deterministic given
``x0``, so beyond "same minimum" we can demand "same trajectory": the
*full simplex* (all n+1 vertices and their fitness values), the simplex
extent / fitness spread (the xatol/fatol test quantities), and the
evaluation counters should track SciPy iteration for iteration.

SciPy's callback only exposes the best vertex, so this experiment reads
the full internal state via **frame introspection**: the per-iteration
callback walks up the call stack to the live ``_minimize_neldermead``
frame and records ``sim`` / ``fsim`` / ``fcalls`` from its locals.  The
oracle is the *installed, unmodified* SciPy (pinned 1.15.x — the frame
locals are version-coupled).

The framework logger fires at the same boundary as SciPy's callback
(after the operation and re-sort), so records align one-to-one by index;
SciPy emits one extra trailing record on convergence (its termination
``break`` still runs the ``finally`` block that fires the callback).

Output (under ``plots/cross_validation/neldermead_vs_scipy/``):

- ``convergence_<func>_d<dim>.png`` — best-fitness-vs-evaluations overlay
- ``state_<func>_d<dim>.png`` — six internal-state trajectories side by side
- ``divergence_<func>_d<dim>.png`` — per-iteration deviation (log scale),
  overlaid with a *chaos sensitivity reference*: the same metrics for
  scipy vs scipy started from ``x0 + 1e-12``.  Nelder-Mead trajectories
  amplify perturbations (sharply, once a discrete accept/reject branch
  flips), so the reference grows by orders of magnitude — which is what
  makes the declivity-vs-scipy line sitting at *exactly zero* a strong
  statement rather than an empty plot
- ``agreement_matrix.png`` — one glanceable grid: max deviation across
  all seeds and iterations per (function x dim) and metric
- ``panels_<func>_d<dim>.png`` — the framework's own diagnostic panels
  (``plot_metrics`` with ``PanelSet.ALL``) for the declivity run
- ``summary.csv`` — one row per (function x dim x seed)

Run::

    PYTHONPATH=. pdm run python experiments/cross_validation/neldermead_vs_scipy.py
"""

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
from declivity.algorithms.neldermead.config import NelderMeadConfig
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

PLOTS_DIR = Path("plots/cross_validation/neldermead_vs_scipy")

XATOL = 1e-4
FATOL = 1e-4


# ---------------------------------------------------------------------------
# SciPy runner with frame-introspection state probe.
# ---------------------------------------------------------------------------


@dataclass
class ScipyNMTrace:
    """Internal state of scipy's ``_minimize_neldermead``, one record per
    iteration, read out of the live solver frame by the callback."""

    evaluations: list[int] = field(default_factory=list)
    sim: list[NDArray[np.float64]] = field(default_factory=list)
    fsim: list[NDArray[np.float64]] = field(default_factory=list)

    # Derived series (filled on demand).

    @property
    def best(self) -> list[float]:
        return [float(f[0]) for f in self.fsim]

    @property
    def worst(self) -> list[float]:
        return [float(f[-1]) for f in self.fsim]

    @property
    def mean(self) -> list[float]:
        return [float(np.mean(f)) for f in self.fsim]

    @property
    def std(self) -> list[float]:
        return [float(np.std(f)) for f in self.fsim]

    @property
    def diameter(self) -> list[float]:
        return [float(np.max(np.abs(s[1:] - s[0]))) for s in self.sim]

    @property
    def spread(self) -> list[float]:
        return [float(np.max(np.abs(f[0] - f[1:]))) for f in self.fsim]

    @property
    def volume(self) -> list[float]:
        out = []
        for s in self.sim:
            n = s.shape[1]
            _, logdet = np.linalg.slogdet(s[1:] - s[0])
            log_factorial = float(np.sum(np.log(np.arange(1, n + 1))))
            out.append(
                float(np.exp(logdet - log_factorial)) if np.isfinite(logdet) else 0.0
            )
        return out


def run_scipy_neldermead(
    func: BenchmarkFunction,
    x0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    budget: int,
):
    """Run unmodified scipy Nelder-Mead, capturing the simplex per iteration."""
    trace = ScipyNMTrace()

    def callback(intermediate_result):
        # Walk up to the _minimize_neldermead frame; its locals hold the
        # full simplex the public API never exposes.
        frame = inspect.currentframe()
        while frame is not None and "sim" not in frame.f_locals:
            frame = frame.f_back
        if frame is None:  # scipy internals changed — record nothing
            return
        state = frame.f_locals
        trace.evaluations.append(int(state["fcalls"][0]))
        trace.sim.append(np.array(state["sim"], dtype=float))
        trace.fsim.append(np.array(state["fsim"], dtype=float))

    result = minimize(
        func,
        np.array(x0, dtype=float),
        method="Nelder-Mead",
        bounds=Bounds(lower, upper),
        options={
            "xatol": XATOL,
            "fatol": FATOL,
            "maxfev": budget,
            "maxiter": 10**9,
        },
        callback=callback,
    )
    return trace, result


# ---------------------------------------------------------------------------
# Framework runner.
# ---------------------------------------------------------------------------


def run_declivity_neldermead(
    func: BenchmarkFunction,
    x0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    budget: int,
):
    config = NelderMeadConfig(dimensions=len(x0), xatol=XATOL, fatol=FATOL)
    config.enable_all_diagnostics()
    optimizer = AlgorithmFactory.create_optimizer(
        AlgorithmChoice.NELDERMEAD,
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

_OURS_STYLE = {"color": "tab:blue", "lw": 1.8, "label": "declivity"}
_SCIPY_STYLE = {"color": "tab:orange", "lw": 1.8, "ls": "--", "label": "scipy"}


def plot_convergence(logs, trace: ScipyNMTrace, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(logs.evaluations, np.maximum(logs.best_fitness, 1e-30), **_OURS_STYLE)
    ax.semilogy(trace.evaluations, np.maximum(trace.best, 1e-30), **_SCIPY_STYLE)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best fitness (log)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_state_trajectories(logs, trace: ScipyNMTrace, title: str, path: Path) -> None:
    """Six internal-state metrics, declivity vs scipy, per iteration."""
    iters_ours = np.asarray(logs.iteration)
    iters_scipy = np.arange(1, len(trace.fsim) + 1)

    panels = [
        ("Best vertex fitness", "log", (logs.best_fitness, trace.best)),
        ("Worst vertex fitness", "log", (logs.worst_fitness, trace.worst)),
        ("Vertex fitness std dev", "log", (logs.std_fitness, trace.std)),
        ("Simplex extent (xatol test)", "log", (logs.simplex_diameter, trace.diameter)),
        ("Fitness spread (fatol test)", "log", (logs.fitness_spread, trace.spread)),
        ("Simplex volume", "log", (logs.simplex_volume, trace.volume)),
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


def compute_diffs(logs, trace: ScipyNMTrace) -> dict[str, float]:
    """Per-iteration deviation metrics between the implementations."""
    n = min(len(logs.iteration), len(trace.fsim))
    if n == 0:
        return {
            "max_f_rel_diff": float("nan"),
            "max_sim_diff": float("nan"),
            "max_fsim_diff": float("nan"),
            "max_evals_diff": -1,
        }
    f_ours = np.asarray(logs.best_fitness[:n])
    f_scipy = np.asarray(trace.best[:n])
    f_diff = np.abs(f_ours - f_scipy) / np.maximum(np.abs(f_scipy), 1e-12)

    sim_diff = np.array(
        [
            float(np.max(np.abs(a - b)))
            for a, b in zip(logs.population[:n], trace.sim[:n])
        ]
    )
    # The logger keeps best/worst per iteration (not the full fsim), so the
    # vertex-fitness diff compares the (best, worst) envelope; the full
    # per-vertex geometry is covered by max |simplex diff| above.
    fsim_diff = np.array(
        [
            float(
                max(
                    abs(logs.best_fitness[i] - trace.fsim[i][0]),
                    abs(logs.worst_fitness[i] - trace.fsim[i][-1]),
                )
            )
            for i in range(n)
        ]
    )
    evals_diff = np.abs(
        np.asarray(logs.evaluations[:n]) - np.asarray(trace.evaluations[:n])
    )
    return {
        "max_f_rel_diff": float(np.max(f_diff)),
        "max_sim_diff": float(np.max(sim_diff)),
        "max_fsim_diff": float(np.max(fsim_diff)),
        "max_evals_diff": int(np.max(evals_diff)),
        "_series": (f_diff, sim_diff, fsim_diff, evals_diff),
    }


def trace_deviation_series(a: ScipyNMTrace, b: ScipyNMTrace) -> tuple[NDArray, NDArray]:
    """Per-iteration deviation between two SciPy traces (used for the
    chaos sensitivity reference: scipy vs scipy with perturbed x0)."""
    n = min(len(a.fsim), len(b.fsim))
    f_diff = np.abs(np.asarray(a.best[:n]) - np.asarray(b.best[:n])) / np.maximum(
        np.abs(np.asarray(b.best[:n])), 1e-12
    )
    sim_diff = np.array(
        [float(np.max(np.abs(p - q))) for p, q in zip(a.sim[:n], b.sim[:n])]
    )
    return f_diff, sim_diff


def plot_divergence(
    logs,
    trace: ScipyNMTrace,
    chaos_reference: tuple[NDArray, NDArray] | None,
    title: str,
    path: Path,
) -> dict:
    """Per-iteration deviation between the two implementations.

    The declivity-vs-scipy deviation is expected to sit at *exactly zero*
    (bit-identical simplices); on its own that renders as an
    uninformative flat line at the log floor.  ``chaos_reference``
    supplies the context that makes the zero meaningful: the same
    metrics for scipy vs scipy started from ``x0 + 1e-12``.  Nelder-Mead
    trajectories amplify perturbations (and diverge sharply the moment a
    discrete accept/reject branch flips), so that seed difference grows
    by orders of magnitude — any real implementation difference would
    show up the same way.
    """
    diffs = compute_diffs(logs, trace)
    f_diff, sim_diff, fsim_diff, evals_diff = diffs.pop("_series")
    floor = 1e-18

    iters = np.arange(1, len(f_diff) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    exact = (
        np.max(f_diff, initial=0) == 0
        and np.max(sim_diff, initial=0) == 0
        and np.max(fsim_diff, initial=0) == 0
        and np.max(evals_diff, initial=0) == 0
    )
    ours_label_suffix = "  (≡ 0, exact)" if exact else ""
    ax.semilogy(
        iters,
        np.maximum(f_diff, floor),
        label=f"declivity vs scipy: |f best diff| (rel){ours_label_suffix}",
        color="tab:blue",
        lw=2.0,
    )
    ax.semilogy(
        iters,
        np.maximum(sim_diff, floor),
        label=f"declivity vs scipy: max |simplex diff|{ours_label_suffix}",
        color="tab:cyan",
        lw=2.0,
        ls="-.",
    )

    if chaos_reference is not None:
        cf, csim = chaos_reference
        c_iters = np.arange(1, len(cf) + 1)
        ax.semilogy(
            c_iters,
            np.maximum(cf, floor),
            label="chaos ref — scipy vs scipy(x0+1e-12): |f best diff| (rel)",
            color="tab:red",
            lw=1.4,
            ls="--",
            alpha=0.8,
        )
        ax.semilogy(
            c_iters,
            np.maximum(csim, floor),
            label="chaos ref — scipy vs scipy(x0+1e-12): max |simplex diff|",
            color="tab:orange",
            lw=1.4,
            ls=":",
            alpha=0.8,
        )

    if exact:
        ax.annotate(
            "declivity ≡ scipy: every metric exactly 0 at every iteration\n"
            "(flat line at the plot floor). The dashed reference shows the same\n"
            "probe detecting a 1e-12 x0 perturbation, which grows by orders of\n"
            "magnitude — the zero is not for lack of sensitivity.",
            xy=(0.02, 0.03),
            xycoords="axes fraction",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.9},
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Deviation (log; floored at 1e-18)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return diffs


def plot_agreement_matrix(summary: pd.DataFrame, path: Path, title: str) -> None:
    """One glanceable figure: max deviation across all seeds and iterations
    per (function x dim) row and metric column.  All-zero cells are the
    headline result — annotated '0 (exact)'."""
    metrics = ["max_f_rel_diff", "max_sim_diff", "max_fsim_diff", "max_evals_diff"]
    labels = [
        "|f best diff| (rel)",
        "max |simplex diff|",
        "max |f(vertex) diff|",
        "|evals diff|",
    ]

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

                scipy_trace, scipy_result = run_scipy_neldermead(
                    fn, x0, lower, upper, budget
                )
                logs, our_result = run_declivity_neldermead(
                    fn, x0, lower, upper, budget
                )

                tag = f"{fn_name}_d{dim}"
                if seed == 0:
                    plot_convergence(
                        logs,
                        scipy_trace,
                        f"Nelder-Mead — {fn_cls.__name__} d={dim} (seed 0)",
                        PLOTS_DIR / f"convergence_{tag}.png",
                    )
                    plot_state_trajectories(
                        logs,
                        scipy_trace,
                        f"Nelder-Mead internal state — {fn_cls.__name__} d={dim} (seed 0)",
                        PLOTS_DIR / f"state_{tag}.png",
                    )
                    plot_metrics(
                        our_result,
                        panels=PanelSet.ALL,
                        title=f"declivity Nelder-Mead — {fn_cls.__name__} d={dim}",
                        save_path=PLOTS_DIR / f"panels_{tag}.png",
                    )
                    plt.close("all")
                    # Chaos sensitivity reference: the same probe applied
                    # to scipy vs scipy started from x0 + 1e-12.
                    scipy_trace_pert, _ = run_scipy_neldermead(
                        fn, x0 + 1e-12, lower, upper, budget
                    )
                    chaos_reference = trace_deviation_series(
                        scipy_trace_pert, scipy_trace
                    )
                    diffs = plot_divergence(
                        logs,
                        scipy_trace,
                        chaos_reference,
                        f"Nelder-Mead deviation — {fn_cls.__name__} d={dim} (seed 0)",
                        PLOTS_DIR / f"divergence_{tag}_s{seed}.png",
                    )
                else:
                    diffs = compute_diffs(logs, scipy_trace)
                    diffs.pop("_series", None)

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
                        "iters_scipy": len(scipy_trace.fsim),
                        **diffs,
                    }
                )
                print(
                    f"[{fn_cls.__name__:10s} d={dim:2d} seed={seed}] "
                    f"f: {our_result.best_fitness:.3e} vs {scipy_result.fun:.3e} | "
                    f"evals: {our_result.evaluations} vs {scipy_result.nfev} | "
                    f"max simplex diff: {diffs['max_sim_diff']:.2e}"
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(PLOTS_DIR / "summary.csv", index=False)
    plot_agreement_matrix(
        summary,
        PLOTS_DIR / "agreement_matrix.png",
        "Nelder-Mead: declivity vs scipy agreement",
    )
    print(f"\nSummary written to {PLOTS_DIR / 'summary.csv'}")
    print(
        summary.groupby(["function", "dim"])[
            ["max_f_rel_diff", "max_sim_diff", "max_fsim_diff", "max_evals_diff"]
        ].max()
    )


if __name__ == "__main__":
    main()
