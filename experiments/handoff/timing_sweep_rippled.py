"""Handoff timing sweep on rotated RippledEllipsoid.

Goal: show how much CMA-ES warmup is needed for the covariance to become
informative. Sweeps the warmup budget while keeping total budget fixed.
Compares C^-1 handoff against identity handoff at each timing.

Use this together with ``rippled_ellipsoid_handoff.py`` to motivate when
the covariance handoff is worth doing.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.lbfgsb import ArmijoBacktracking
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.benchmarking import (
    Benchmark,
    CMAESLBFGSBHandoff,
    Problem,
    SingleAlgorithm,
)
from declivity.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from declivity.utils.benchmark_functions import RippledEllipsoid, RotatedFunction


plt.ioff()
plt.switch_backend("Agg")


PALETTE_INVERSE = ["#a8e6c9", "#74c69d", "#40916c", "#2d6a4f", "#1b4332"]
PALETTE_IDENTITY = ["#dcd0ff", "#b39ddb", "#9575cd", "#7e57c2", "#5e35b1"]

REFERENCE_COLORS = {
    "CMA-ES":   "#e74c3c",
    "L-BFGS-B": "#3498db",
}


def build_problem(d: int, condition: float, amplitude: float, rotation_seed: int):
    base = RippledEllipsoid(d, condition=condition, amplitude=amplitude)
    rotated = RotatedFunction(base, rotation="random", seed=rotation_seed)
    return Problem.from_benchmark(
        f"RippledEllipsoid-c{int(condition)}-a{amplitude:g}-d{d}", rotated
    )


def build_algorithms(
    total_budget: int,
    warmup_budgets: list[int],
    memory_size: int,
    initial_sigma: float,
):
    lbfgsb_kwargs = dict(m=memory_size, pgtol=1e-10, factr=0)
    armijo = ArmijoBacktracking()

    cmaes_only = SingleAlgorithm(
        name="CMA-ES",
        color=REFERENCE_COLORS["CMA-ES"],
        algorithm=AlgorithmChoice.CMAES,
        config_factory=lambda d: CMAESConfig(
            dimensions=d, budget=total_budget, sigma=initial_sigma,
        ),
    )

    lbfgsb_only = SingleAlgorithm(
        name="L-BFGS-B",
        color=REFERENCE_COLORS["L-BFGS-B"],
        algorithm=AlgorithmChoice.LBFGSB,
        config_factory=lambda d: LBFGSBConfig(
            dimensions=d, budget=total_budget, **lbfgsb_kwargs,
        ),
        line_search=armijo,
    )

    handoffs = []
    for idx, warmup in enumerate(warmup_budgets):
        post = total_budget - warmup
        # C^-1
        handoffs.append(
            CMAESLBFGSBHandoff(
                name=f"C^-1 warmup={warmup}",
                color=PALETTE_INVERSE[idx % len(PALETTE_INVERSE)],
                cmaes_config_factory=lambda d, w=warmup: CMAESConfig(
                    dimensions=d, budget=w, sigma=initial_sigma,
                ),
                lbfgsb_config_factory=lambda d, p=post: LBFGSBConfig(
                    dimensions=d, budget=p, **lbfgsb_kwargs,
                ),
                transform="inverse",
                lbfgsb_line_search=armijo,
            )
        )
        # Identity
        handoffs.append(
            CMAESLBFGSBHandoff(
                name=f"identity warmup={warmup}",
                color=PALETTE_IDENTITY[idx % len(PALETTE_IDENTITY)],
                cmaes_config_factory=lambda d, w=warmup: CMAESConfig(
                    dimensions=d, budget=w, sigma=initial_sigma,
                ),
                lbfgsb_config_factory=lambda d, p=post: LBFGSBConfig(
                    dimensions=d, budget=p, **lbfgsb_kwargs,
                ),
                transform="identity",
                lbfgsb_line_search=armijo,
            )
        )
    return [cmaes_only, lbfgsb_only, *handoffs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=30)
    parser.add_argument("--condition", type=float, default=1e6)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--memory-size", type=int, default=5)
    parser.add_argument("--total-budget", type=int, default=10000)
    parser.add_argument(
        "--warmup-budgets", type=int, nargs="+",
        default=[500, 1500, 3000, 5000, 7500],
    )
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--rotation-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("plots/handoff/timing_sweep_rippled"),
    )
    args = parser.parse_args()

    problem = build_problem(
        args.dimensions, args.condition, args.amplitude, args.rotation_seed
    )

    algorithms = build_algorithms(
        total_budget=args.total_budget,
        warmup_budgets=sorted(args.warmup_budgets),
        memory_size=args.memory_size,
        initial_sigma=2.0,
    )

    bench = Benchmark(
        problems=[problem],
        algorithms=algorithms,
        seeds=list(range(args.num_seeds)),
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
    bench.run(verbose=True)
    bench.print_summary()

    plot_benchmark_convergence(
        bench.traces,
        problems=[problem],
        algorithms=algorithms,
        save_path=args.output_dir / "convergence.png",
        title=(
            f"Handoff timing on RippledEllipsoid (cond={args.condition:.0e}, "
            f"amp={args.amplitude}, d={args.dimensions}, m={args.memory_size}, "
            f"{args.num_seeds} seeds)"
        ),
        show_iqr=False,
    )
    plot_benchmark_boxplot(
        bench.traces,
        problems=[problem],
        algorithms=algorithms,
        save_path=args.output_dir / "final_fitness.png",
        title=f"Final fitness vs warmup timing",
    )

    print(f"\nPlots saved to {args.output_dir.absolute()}/")


if __name__ == "__main__":
    main()
