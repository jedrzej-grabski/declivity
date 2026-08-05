"""Render a single composite parity-audit figure across a problem battery.

This script reuses the ``_run_framework`` and ``_run_reference``
helpers from :mod:`des_vs_reference` and :mod:`mfcmaes_vs_reference`
and lays the per-seed convergence overlays out as a grid: one row per
algorithm (DES, MF-CMA-ES) and one column per problem.  The result is
one PNG that visualises distributional parity across multiple problem
classes at a glance.

Output (under ``plots/cross_validation/parity_grid/``):

- ``parity_grid_<problems>.png`` — the composite convergence figure
- ``parity_grid_<problems>.csv`` — Wilcoxon / KS p-values per
  (algorithm × problem) cell

Run::

    PYTHONPATH=. pdm run python -m experiments.cross_validation.parity_grid
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiments.cross_validation import des_vs_reference as des_h
from experiments.cross_validation import mfcmaes_vs_reference as mf_h
from experiments.cross_validation._problems import PROBLEMS, ProblemSpec

DEFAULT_PROBLEMS = ("ellipsoid_d10", "rosenbrock_d10", "rastrigin_d10")
DEFAULT_SEEDS = 15
DEFAULT_OUTPUT_DIR = Path("plots/cross_validation/parity_grid")


@dataclass
class _Algo:
    name: str
    framework_runner: Callable
    reference_runner: Callable
    legend_label_port: str


ALGOS = [
    _Algo(
        name="DES",
        framework_runner=des_h._run_framework,
        reference_runner=des_h._run_reference,
        legend_label_port="port DES.R",
    ),
    _Algo(
        name="MF-CMA-ES",
        framework_runner=mf_h._run_framework,
        reference_runner=mf_h._run_reference,
        legend_label_port="port nm_cma_es",
    ),
]


def _draw_panel(
    ax, spec: ProblemSpec, fw_runs, ref_runs, algo: _Algo, p_w: float, p_ks: float
) -> None:
    for r in fw_runs:
        if not r.iter_best:
            continue
        ax.plot(
            r.iter_evals,
            np.maximum(r.iter_best, spec.floor),
            color="C0",
            alpha=0.4,
            lw=0.9,
        )
    for r in ref_runs:
        if not r.iter_best:
            continue
        ax.plot(
            r.iter_evals,
            np.maximum(r.iter_best, spec.floor),
            color="C3",
            alpha=0.4,
            lw=0.9,
            linestyle="--",
        )

    if spec.f_star > 0:
        ax.axhline(spec.f_star, color="gray", lw=0.7, linestyle=":", alpha=0.6)

    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        f"{algo.name} · {spec.name}    W p={p_w:.2f}  KS p={p_ks:.2f}",
        fontsize=10,
    )


def run(problem_keys: list[str], seeds: list[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(ALGOS)
    n_cols = len(problem_keys)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.4 * n_rows),
        sharex=False,
        sharey=False,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    rows: list[dict] = []

    for i, algo in enumerate(ALGOS):
        for j, problem_key in enumerate(problem_keys):
            spec = PROBLEMS[problem_key]
            print(
                f"[{algo.name} · {spec.name}] running {len(seeds)} seeds…", flush=True
            )
            fw_runs = [algo.framework_runner(spec, s) for s in seeds]
            ref_runs = [algo.reference_runner(spec, s) for s in seeds]

            fw_finals = np.array([r.best_fit for r in fw_runs])
            ref_finals = np.array([r.best_fit for r in ref_runs])
            try:
                _w_stat, p_w = stats.wilcoxon(
                    fw_finals, ref_finals, zero_method="zsplit"
                )
            except ValueError:
                _w_stat, p_w = 0.0, 1.0  # all pairs equal — treat as match
            ks_stat, p_ks = stats.ks_2samp(fw_finals, ref_finals)

            ax = axes[i, j]
            _draw_panel(ax, spec, fw_runs, ref_runs, algo, float(p_w), float(p_ks))

            if i == n_rows - 1:
                ax.set_xlabel("liczba ewaluacji")
            if j == 0:
                ax.set_ylabel("najlepsza wartość f")

            rows.append(
                {
                    "algorithm": algo.name,
                    "problem": spec.name,
                    "fw_median": float(np.median(fw_finals)),
                    "ref_median": float(np.median(ref_finals)),
                    "fw_std": float(fw_finals.std(ddof=1)),
                    "ref_std": float(ref_finals.std(ddof=1)),
                    "wilcoxon_p": float(p_w),
                    "ks_p": float(p_ks),
                    "n_seeds": len(seeds),
                }
            )
            print(
                f"  {algo.name:<10} {spec.name:<12} "
                f"fw_med={np.median(fw_finals):.3e}  ref_med={np.median(ref_finals):.3e}  "
                f"W p={p_w:.3f}  KS p={p_ks:.3f}"
            )

    # Shared legend (one for the whole figure).
    legend_handles = [
        plt.Line2D([], [], color="C0", lw=2, label="declivity"),
        plt.Line2D([], [], color="C3", lw=2, linestyle="--", label="port referencyjny"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    fig.suptitle("", y=1.0)  # leave room for the legend
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    tag = "_".join(PROBLEMS[k].name for k in problem_keys)
    fig_path = output_dir / f"parity_grid_{tag}.png"
    csv_path = output_dir / f"parity_grid_{tag}.csv"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nwrote {fig_path}")
    print(f"wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(DEFAULT_PROBLEMS),
        choices=sorted(PROBLEMS.keys()),
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    run(problem_keys=args.problems, seeds=seeds, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
