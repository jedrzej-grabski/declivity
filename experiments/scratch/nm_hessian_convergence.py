#!/usr/bin/env python3
"""Does seeding Nelder-Mead with the Hessian actually improve convergence?

Companion to ``nm_hessian_2d.py``, which can only ever answer "no": at d = 2 a
simplex has 3 vertices and Nelder-Mead repairs its own shape within a handful
of reflections, so every conditioner converges at the same rate and the curves
land on top of each other.  The cost of a badly shaped simplex grows with
dimension, so this runs the same three arms across d and plots the multi-seed
median + IQR convergence band per dimension.

Arms (all sharing x0, rotation, budget and NM config per seed):

    I  @ 0.1*span      exp1's isotropic control
    H  @ 0.1*span      exp1's H^-1 arm: right shape, size measured in x-space
    H  @ newton        right shape, size measured in the metric H defines --
                       whitened side sqrt(g' H^-1 g), the Newton step length

Throwaway; nothing here extends the framework.

    PYTHONPATH=. uv run python experiments/scratch/nm_hessian_convergence.py
    PYTHONPATH=. uv run python experiments/scratch/nm_hessian_convergence.py \
        --dims 10,30 --num-seeds 25

Writes ``plots/nm2d/convergence/convergence_by_dimension.png`` plus one PNG
per dimension.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.neldermead.config import NelderMeadConfig
from declivity.algorithms.neldermead.neldermead_optimizer import NelderMeadOptimizer
from declivity.benchmarking.problem import Problem
from declivity.utils.benchmark_functions import Ellipsoid, RotatedFunction
from declivity.utils.hessian import numerical_hessian, spd_regularize
from declivity.utils.initial_geometry import InitialGeometry
from declivity.utils.initial_point_generator import UniformBoxInitialPointGenerator
from declivity.utils.population_initializers import CovarianceSimplexInitializer
from declivity.utils.stopping_conditions import MaxEvaluations

SAMPLING_LOWER, SAMPLING_UPPER = -100.0, 100.0
BASE_SIZE = 0.1 * (SAMPLING_UPPER - SAMPLING_LOWER)
BUDGET_PER_DIM = 500
GAP_FLOOR = 1e-16
TARGET = 1e-8

IDENTITY_COLOR = "#f5a623"
HESSIAN_COLOR = "#ffffff"
NEWTON_COLOR = "#63e6be"


@dataclass
class Curve:
    evaluations: NDArray[np.float64]
    gaps: NDArray[np.float64]


def rotation_matrix(dim: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng([911, dim, seed])
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


def make_problem(dim: int, seed: int) -> Problem:
    function = RotatedFunction(
        Ellipsoid(dim, lower=SAMPLING_LOWER, upper=SAMPLING_UPPER),
        rotation=rotation_matrix(dim, seed),
        name_suffix=f"seed{seed}",
    )
    return Problem(
        name=f"Ellipsoid-d{dim}",
        function=function,
        dimensions=dim,
        lower_bound=-np.inf,
        upper_bound=np.inf,
        initial_point_generator=UniformBoxInitialPointGenerator(
            SAMPLING_LOWER, SAMPLING_UPPER
        ),
    )


def fd_gradient(func, x: NDArray[np.float64], eps: float = 1e-5) -> NDArray[np.float64]:
    gradient = np.empty_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps * max(1.0, abs(float(x[i])))
        gradient[i] = (func(x + step) - func(x - step)) / (2.0 * step[i])
    return gradient


def run(problem: Problem, x0, initializer, dim: int, budget: int, seed: int) -> Curve:
    config = NelderMeadConfig(dimensions=dim, xatol=1e-10, fatol=1e-12, adaptive=True)
    optimizer = NelderMeadOptimizer(
        func=problem.function,
        initial_point=x0,
        config=config,
        population_initializer=initializer,
        constraint_handler=problem.resolved_constraint_handler(),
        stopping_condition=MaxEvaluations(budget),
        lower_bounds=problem.lower_bound,
        upper_bounds=problem.upper_bound,
        seed=seed,
    )
    logs = optimizer.optimize().diagnostic
    return Curve(
        evaluations=np.asarray(logs.evaluations, dtype=float),
        gaps=np.maximum(np.asarray(logs.best_fitness, dtype=float), GAP_FLOOR),
    )


def band(
    curves: list[Curve], grid: NDArray[np.float64]
) -> tuple[NDArray, NDArray, NDArray]:
    """Median and IQR of the step-interpolated curves on a common eval grid.

    A run that stopped early (internal convergence) holds its final value, so
    the band stays defined out to the full budget.
    """
    stacked = np.empty((len(curves), grid.size))
    for row, curve in enumerate(curves):
        index = np.searchsorted(curve.evaluations, grid, side="right") - 1
        stacked[row] = np.where(index >= 0, curve.gaps[np.maximum(index, 0)], np.nan)
    return (
        np.nanmedian(stacked, axis=0),
        np.nanpercentile(stacked, 25, axis=0),
        np.nanpercentile(stacked, 75, axis=0),
    )


def evals_to_target(curve: Curve, budget: int) -> int:
    hit = np.flatnonzero(curve.gaps < TARGET)
    return int(curve.evaluations[hit[0]]) if hit.size else budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", default="2,5,10,30")
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--out", type=Path, default=Path("plots/nm2d/convergence"))
    args = parser.parse_args()

    dims = [int(token) for token in args.dims.split(",") if token.strip()]
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#111111",
            "axes.facecolor": "#111111",
            "savefig.facecolor": "#111111",
            "grid.alpha": 0.25,
            "legend.framealpha": 0.35,
            "font.size": 9,
        }
    )
    plt.switch_backend("Agg")
    args.out.mkdir(parents=True, exist_ok=True)

    columns = min(len(dims), 2)
    rows = (len(dims) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.4 * columns, 4.6 * rows), squeeze=False
    )

    for position, dim in enumerate(dims):
        budget = BUDGET_PER_DIM * dim
        results: dict[str, list[Curve]] = {"i": [], "h": [], "n": []}

        for seed in range(args.num_seeds):
            problem = make_problem(dim, seed)
            x0 = problem.starting_point(seed)
            geometry_h = InitialGeometry.from_curvature(
                spd_regularize(numerical_hessian(problem.function, x0)), dim
            )
            geometry_i = InitialGeometry.identity(dim)
            gradient = fd_gradient(problem.function, x0)
            alpha = float(np.sqrt(max(gradient @ geometry_h.solve(gradient), 0.0)))

            results["i"].append(
                run(
                    problem,
                    x0,
                    CovarianceSimplexInitializer(
                        geometry_i,
                        base_size=BASE_SIZE,
                        min_step=1e-8,
                        ratio_floor=1e-12,
                    ),
                    dim,
                    budget,
                    seed,
                )
            )
            results["h"].append(
                run(
                    problem,
                    x0,
                    CovarianceSimplexInitializer(
                        geometry_h,
                        base_size=BASE_SIZE,
                        min_step=1e-8,
                        ratio_floor=1e-12,
                    ),
                    dim,
                    budget,
                    seed,
                )
            )
            results["n"].append(
                run(
                    problem,
                    x0,
                    CovarianceSimplexInitializer(
                        geometry_h,
                        base_size=alpha,
                        normalize=False,
                        min_step=1e-8,
                        ratio_floor=1e-12,
                    ),
                    dim,
                    budget,
                    seed,
                )
            )

        # Start the grid where every arm has already logged its first
        # iteration, so no grid point sits before all of the curves.
        start = max(
            float(curve.evaluations[0])
            for curves in results.values()
            for curve in curves
        )
        grid = np.unique(np.geomspace(start, budget, 500).astype(int)).astype(float)
        ax = axes[position // columns][position % columns]
        summary = []
        for key, label, color in (
            ("i", "$I$ @ 0.1·span", IDENTITY_COLOR),
            ("h", "$H^{-1}$ @ 0.1·span  (exp1)", HESSIAN_COLOR),
            ("n", "$H^{-1}$ @ newton", NEWTON_COLOR),
        ):
            median, low, high = band(results[key], grid)
            ax.plot(grid, median, "-", color=color, lw=1.8, label=label)
            ax.fill_between(grid, low, high, color=color, alpha=0.18, linewidth=0)
            reached = [evals_to_target(curve, budget) for curve in results[key]]
            solved = sum(1 for curve in results[key] if curve.gaps[-1] < TARGET)
            summary.append(
                f"{label}: {int(np.median(reached))} evals, {solved}/{args.num_seeds} solved"
            )

        ax.axhline(TARGET, color="#888888", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("evaluations")
        ax.set_ylabel("$f(x_{best}) - f^*$")
        ax.set_title(
            f"rotated Ellipsoid, d={dim}  (budget {budget}, {args.num_seeds} seeds)",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=8)
        print(f"d={dim}: " + "  |  ".join(summary))

        single = plt.figure(figsize=(7.0, 5.0))
        single_ax = single.add_subplot(111)
        for line in ax.get_lines():
            single_ax.plot(
                *line.get_data(),
                color=line.get_color(),
                lw=line.get_linewidth(),
                ls=line.get_linestyle(),
                label=line.get_label(),
            )
        single_ax.set(
            xscale="log",
            yscale="log",
            xlabel="evaluations",
            ylabel="$f(x_{best}) - f^*$",
        )
        single_ax.set_title(ax.get_title(), fontsize=10)
        single_ax.grid(True, alpha=0.25)
        single_ax.legend(loc="lower left", fontsize=8)
        single.tight_layout()
        single.savefig(args.out / f"convergence_d{dim:03d}.png", dpi=130)
        plt.close(single)

    for extra in range(len(dims), rows * columns):
        axes[extra // columns][extra % columns].axis("off")
    fig.suptitle(
        "Nelder-Mead: does seeding the simplex with $H^{-1}$ help?  "
        "median + IQR over seeds",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = args.out / "convergence_by_dimension.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
