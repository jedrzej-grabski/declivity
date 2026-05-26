"""Multimodal handoff benchmark: CMA-ES + L-BFGS-B vs each alone.

Compares four algorithms on Rastrigin and Griewank by default:

1. CMA-ES standalone, full budget.
2. L-BFGS-B standalone, full budget.
3. CMA-ES warm-up -> L-BFGS-B with C^{-1} initial Hessian (the headline
   handoff).
4. CMA-ES warm-up -> L-BFGS-B with identity B_0 (same x0 as the handoff
   but no covariance information passed; isolates "does the covariance
   actually help" from "does the CMA-ES warmup find a better basin").

Same seed produces the same starting point for all algorithms, and the
same CMA-ES RNG path, so the convergence curves of (1), (3), and (4)
match each other up to their respective handoff points.

The total evaluation budget is fixed across all algorithms, so the
comparison is fair on the dimension the user cares about most: function
evaluations.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.lbfgsb.config import LBFGSBConfig, LineSearchMethod
from src.benchmarking import (
    Benchmark,
    BenchmarkPlotter,
    CMAESLBFGSBHandoff,
    Problem,
    SingleAlgorithm,
)
from src.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from src.utils.benchmark_functions import Griewank, Rastrigin


plt.ioff()
plt.switch_backend("Agg")


# MIGRATION COMPARE — populated alongside the production output for
# side-by-side review of the legacy and declarative plotters. Remove this
# constant and the corresponding write paths once the new plots are signed
# off on.
COMPARE_DIR = Path("plots/_migration_compare/multimodal")


COLORS = {
    "CMA-ES":                          "#e74c3c",
    "L-BFGS-B":                        "#3498db",
    "CMA-ES -> L-BFGS-B (C^-1)":       "#2ecc71",
    "CMA-ES -> L-BFGS-B (identity)":   "#9b59b6",
}


def build_algorithms(
    total_budget: int,
    cmaes_warmup_budget: int,
    initial_sigma: float,
    memory_size: int,
    include_identity_baseline: bool,
):
    """Algorithms sharing one total evaluation budget.

    Handoffs split the budget: ``cmaes_warmup_budget`` evaluations on
    CMA-ES, the rest on L-BFGS-B.
    """
    lbfgsb_handoff_budget = total_budget - cmaes_warmup_budget

    cmaes_only = SingleAlgorithm(
        name="CMA-ES",
        color=COLORS["CMA-ES"],
        algorithm=AlgorithmChoice.CMAES,
        config_factory=lambda d: CMAESConfig(
            dimensions=d,
            budget=total_budget,
            sigma=initial_sigma,
        ),
    )

    # ARMIJO (simple backtracking) handles the cosine ripples of multimodal
    # functions much better than More-Thuente: the strong-Wolfe curvature
    # condition rejects valid descent steps when the gradient wiggles, so
    # More-Thuente bails out within ~20 evals near the basin floor while
    # Armijo descends steadily. factr=0 disables function-value tolerance.
    lbfgsb_only = SingleAlgorithm(
        name="L-BFGS-B",
        color=COLORS["L-BFGS-B"],
        algorithm=AlgorithmChoice.LBFGSB,
        config_factory=lambda d: LBFGSBConfig(
            dimensions=d,
            budget=total_budget,
            m=memory_size,
            pgtol=1e-10,
            factr=0,
            line_search=LineSearchMethod.ARMIJO,
        ),
    )

    handoff_inverse = CMAESLBFGSBHandoff(
        name="CMA-ES -> L-BFGS-B (C^-1)",
        color=COLORS["CMA-ES -> L-BFGS-B (C^-1)"],
        cmaes_config_factory=lambda d: CMAESConfig(
            dimensions=d,
            budget=cmaes_warmup_budget,
            sigma=initial_sigma,
        ),
        lbfgsb_config_factory=lambda d: LBFGSBConfig(
            dimensions=d,
            budget=lbfgsb_handoff_budget,
            m=memory_size,
            pgtol=1e-10,
            factr=0,
            line_search=LineSearchMethod.ARMIJO,
        ),
        transform="inverse",
    )

    algorithms = [cmaes_only, lbfgsb_only, handoff_inverse]

    if include_identity_baseline:
        handoff_identity = CMAESLBFGSBHandoff(
            name="CMA-ES -> L-BFGS-B (identity)",
            color=COLORS["CMA-ES -> L-BFGS-B (identity)"],
            cmaes_config_factory=lambda d: CMAESConfig(
                dimensions=d,
                budget=cmaes_warmup_budget,
                sigma=initial_sigma,
            ),
            lbfgsb_config_factory=lambda d: LBFGSBConfig(
                dimensions=d,
                budget=lbfgsb_handoff_budget,
                m=memory_size,
                pgtol=1e-10,
                factr=0,
                line_search=LineSearchMethod.ARMIJO,
            ),
            transform="identity",
        )
        algorithms.append(handoff_identity)

    return algorithms


def run_multimodal_handoff(
    dimensions: int,
    total_budget: int,
    cmaes_warmup_budget: int,
    num_seeds: int,
    initial_sigma_rastrigin: float,
    initial_sigma_griewank: float,
    memory_size: int,
    num_workers: int,
    include_identity_baseline: bool,
    output_dir: Path,
) -> None:
    """Two separate sub-benchmarks (one per problem) so each can use its own
    initial sigma; otherwise the framework would force a single sigma across
    both Rastrigin (small box [-5.12, 5.12]) and Griewank (big box [-600, 600])."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(num_seeds))

    sub_settings = [
        ("Rastrigin", Rastrigin(dimensions), initial_sigma_rastrigin),
        ("Griewank",  Griewank(dimensions),  initial_sigma_griewank),
    ]

    all_problems: list[Problem] = []
    all_algorithms = build_algorithms(
        total_budget, cmaes_warmup_budget, initial_sigma_rastrigin, memory_size,
        include_identity_baseline,
    )
    combined_traces: dict = {}

    for problem_name, function, initial_sigma in sub_settings:
        problem = Problem.from_benchmark(problem_name, function)
        algorithms = build_algorithms(
            total_budget, cmaes_warmup_budget, initial_sigma, memory_size,
            include_identity_baseline,
        )

        bench = Benchmark(
            problems=[problem],
            algorithms=algorithms,
            seeds=seeds,
            output_dir=output_dir / problem_name.lower(),
            num_workers=num_workers,
        )
        bench.run(verbose=True)
        bench.print_summary()

        all_problems.append(problem)
        for key, traces in bench.traces.items():
            combined_traces[key] = traces

    convergence_title = (
        f"Multimodal handoff: CMA-ES vs L-BFGS-B vs combined "
        f"({dimensions}D, {num_seeds} seeds, total budget {total_budget})"
    )
    final_fitness_title = (
        f"Final fitness distribution ({dimensions}D, {num_seeds} seeds)"
    )

    # Production output (new plotter).
    plot_benchmark_convergence(
        combined_traces,
        problems=all_problems,
        algorithms=all_algorithms,
        title=convergence_title,
        save_path=output_dir / "convergence.png",
    )
    plot_benchmark_boxplot(
        combined_traces,
        problems=all_problems,
        algorithms=all_algorithms,
        title=final_fitness_title,
        save_path=output_dir / "final_fitness.png",
    )

    # MIGRATION COMPARE — duplicate the same plots into the review dir,
    # alongside the legacy BenchmarkPlotter output. Remove this block once
    # the new plots are signed off.
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    plot_benchmark_convergence(
        combined_traces,
        problems=all_problems,
        algorithms=all_algorithms,
        title=convergence_title,
        save_path=COMPARE_DIR / "new__convergence.png",
    )
    plot_benchmark_boxplot(
        combined_traces,
        problems=all_problems,
        algorithms=all_algorithms,
        title=final_fitness_title,
        save_path=COMPARE_DIR / "new__final_fitness.png",
    )
    legacy_plotter = BenchmarkPlotter(
        problems=all_problems,
        algorithms=all_algorithms,
        traces=combined_traces,
        output_dir=COMPARE_DIR,
    )
    legacy_plotter.plot_convergence_grid(
        save_path=COMPARE_DIR / "old__convergence.png",
        title=convergence_title,
    )
    legacy_plotter.plot_final_fitness_boxplot(
        save_path=COMPARE_DIR / "old__final_fitness.png",
        title=final_fitness_title,
    )

    print(f"\nPlots saved to: {output_dir.absolute()}")
    print(f"Compare plots:  {COMPARE_DIR.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--total-budget", type=int, default=10000)
    parser.add_argument(
        "--cmaes-warmup-budget", type=int, default=1500,
        help=(
            "Evaluations spent on CMA-ES warm-up before the handoff. "
            "Too high and CMA-ES converges fully and L-BFGS-B has nothing "
            "to refine; too low and CMA-ES hasn't found a basin yet."
        ),
    )
    parser.add_argument("--num-seeds", type=int, default=25)
    parser.add_argument("--memory-size", type=int, default=10)
    parser.add_argument(
        "--sigma-rastrigin", type=float, default=2.0,
        help="Initial sigma for CMA-ES on Rastrigin (bounds [-5.12, 5.12]).",
    )
    parser.add_argument(
        "--sigma-griewank", type=float, default=200.0,
        help="Initial sigma for CMA-ES on Griewank (bounds [-600, 600]).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Process count for parallel run execution (>1 uses joblib loky backend).",
    )
    parser.add_argument(
        "--include-identity-baseline", action="store_true",
        help=(
            "Add a 4th algorithm: CMA-ES warmup then L-BFGS-B with identity "
            "B_0. Isolates the value of *passing covariance information* "
            "from the value of *sharing a starting point*."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("plots/handoff/multimodal"),
    )
    args = parser.parse_args()

    if args.cmaes_warmup_budget >= args.total_budget:
        raise ValueError(
            "cmaes-warmup-budget must be strictly less than total-budget"
        )

    run_multimodal_handoff(
        dimensions=args.dimensions,
        total_budget=args.total_budget,
        cmaes_warmup_budget=args.cmaes_warmup_budget,
        num_seeds=args.num_seeds,
        initial_sigma_rastrigin=args.sigma_rastrigin,
        initial_sigma_griewank=args.sigma_griewank,
        memory_size=args.memory_size,
        num_workers=args.num_workers,
        include_identity_baseline=args.include_identity_baseline,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
