"""Handoff timing sweep on multimodal problems.

Same comparison as ``multimodal_handoff_benchmark.py`` but with a family
of handoff timings instead of a single one. Five handoffs are run alongside
CMA-ES standalone and L-BFGS-B standalone, all under the same total budget.

Because every run shares the same x0 (deterministic from the seed) AND
the same CMA-ES RNG, all handoff curves trace the *same* CMA-ES path
until their handoff point — the median plot shows them forking off one
by one as their respective L-BFGS-B refinement kicks in.

Reads as: "would handing off here have been better?".

Timing can be expressed in *evaluations* (default) or *iterations* (CMA-ES
generations) via ``--timing-unit``. Iterations are converted to evaluations
using the CMA-ES population size: evals = generations * (pop_size + 1).
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
from declivity.plotting import plot_benchmark_convergence
from declivity.utils.benchmark_functions import Griewank, Rastrigin
from declivity.utils.stopping_conditions import MaxEvaluations

plt.ioff()
plt.switch_backend("Agg")


# Sequential green palette for handoff timings — earliest handoff is the
# lightest, latest is the darkest. CMA-ES and L-BFGS-B keep their
# distinctive colours so they stand out as references.
HANDOFF_PALETTE = [
    "#a8e6c9",  # earliest
    "#74c69d",
    "#40916c",
    "#2d6a4f",
    "#1b4332",  # latest
]

REFERENCE_COLORS = {
    "CMA-ES": "#e74c3c",
    "L-BFGS-B": "#3498db",
}


def cmaes_evals_per_generation(dimensions: int) -> int:
    """CMA-ES costs pop_size + 1 evaluations per generation (the +1 is the
    mean evaluation logged each iteration). pop_size derives from the
    default CMAESConfig given the problem dimension."""
    return CMAESConfig(dimensions=dimensions).population_size + 1


def resolve_warmup_budgets(
    timings: list[int],
    timing_unit: str,
    dimensions: int,
) -> list[tuple[int, str]]:
    """Translate user-facing timings into (warmup_evals, label).

    label is what shows up on plot legends — keeps the user's chosen unit
    visible.
    """
    if timing_unit == "evaluations":
        return [(t, f"@ {t} evals") for t in timings]
    if timing_unit == "iterations":
        evals_per_gen = cmaes_evals_per_generation(dimensions)
        return [
            (t * evals_per_gen, f"@ {t} gen ({t * evals_per_gen} evals)")
            for t in timings
        ]
    raise ValueError(
        f"Unknown timing-unit {timing_unit!r}; use 'evaluations' or 'iterations'."
    )


def build_algorithms(
    total_budget: int,
    warmup_pairs: list[tuple[int, str]],
    initial_sigma: float,
    memory_size: int,
):
    """One CMA-ES standalone + one L-BFGS-B standalone + one handoff per timing.

    Each handoff splits the total budget so its (CMA-ES warmup) +
    (L-BFGS-B post-handoff) budgets sum to ``total_budget``.
    """
    cmaes_only = SingleAlgorithm(
        name="CMA-ES",
        color=REFERENCE_COLORS["CMA-ES"],
        algorithm=AlgorithmChoice.CMAES,
        config_factory=lambda d: CMAESConfig(
            dimensions=d,
            sigma=initial_sigma,
        ),
        stopping_condition=MaxEvaluations(total_budget),
    )

    armijo = ArmijoBacktracking()

    lbfgsb_only = SingleAlgorithm(
        name="L-BFGS-B",
        color=REFERENCE_COLORS["L-BFGS-B"],
        algorithm=AlgorithmChoice.LBFGSB,
        config_factory=lambda d: LBFGSBConfig(
            dimensions=d,
            m=memory_size,
            pgtol=1e-10,
            factr=0,
        ),
        line_search=armijo,
        stopping_condition=MaxEvaluations(total_budget),
    )

    handoffs: list[CMAESLBFGSBHandoff] = []
    for idx, (warmup_budget, label_suffix) in enumerate(warmup_pairs):
        post_handoff_budget = total_budget - warmup_budget
        color = HANDOFF_PALETTE[idx % len(HANDOFF_PALETTE)]
        handoffs.append(
            CMAESLBFGSBHandoff(
                name=f"Handoff {label_suffix}",
                color=color,
                cmaes_config_factory=lambda d: CMAESConfig(
                    dimensions=d,
                    sigma=initial_sigma,
                ),
                lbfgsb_config_factory=lambda d: LBFGSBConfig(
                    dimensions=d,
                    m=memory_size,
                    pgtol=1e-10,
                    factr=0,
                ),
                transform="inverse",
                lbfgsb_line_search=armijo,
                cmaes_stopping_condition=MaxEvaluations(warmup_budget),
                lbfgsb_stopping_condition=MaxEvaluations(post_handoff_budget),
            )
        )

    return [cmaes_only, lbfgsb_only, *handoffs]


def run_handoff_timing_sweep(
    dimensions: int,
    total_budget: int,
    timings: list[int],
    timing_unit: str,
    num_seeds: int,
    initial_sigma_rastrigin: float,
    initial_sigma_griewank: float,
    memory_size: int,
    num_workers: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(num_seeds))

    warmup_pairs = resolve_warmup_budgets(timings, timing_unit, dimensions)
    print(f"Timing unit: {timing_unit}")
    for raw, (evals, label) in zip(timings, warmup_pairs):
        print(f"  {raw} {timing_unit} -> {evals} evals  (label: 'Handoff {label}')")

    sub_settings = [
        ("Rastrigin", Rastrigin(dimensions), initial_sigma_rastrigin),
        ("Griewank", Griewank(dimensions), initial_sigma_griewank),
    ]

    all_problems: list[Problem] = []
    combined_traces: dict = {}

    # One representative algorithm list (Griewank's sigma) for the plotter;
    # algorithm metadata (name, color) is the same across sigmas so this is
    # safe to use for plotting both panels.
    representative_algorithms = build_algorithms(
        total_budget,
        warmup_pairs,
        initial_sigma_griewank,
        memory_size,
    )

    for problem_name, function, initial_sigma in sub_settings:
        problem = Problem.from_benchmark(problem_name, function)
        algorithms = build_algorithms(
            total_budget,
            warmup_pairs,
            initial_sigma,
            memory_size,
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

    # IQR bands turned off — with 7 algorithms in one panel they would
    # smother the median lines.
    plot_benchmark_convergence(
        combined_traces,
        problems=all_problems,
        algorithms=representative_algorithms,
        save_path=output_dir / "handoff_timing_convergence.png",
        title=(
            f"Handoff timing sweep: when to switch from CMA-ES to L-BFGS-B "
            f"({dimensions}D, {num_seeds} seeds, total budget {total_budget})"
        ),
        show_iqr=False,
    )

    print(
        f"\nPlot saved to: {(output_dir / 'handoff_timing_convergence.png').absolute()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--total-budget", type=int, default=10000)
    parser.add_argument(
        "--timing-unit",
        choices=("evaluations", "iterations"),
        default="evaluations",
        help=(
            "Whether --handoff-timings is given in CMA-ES function evaluations "
            "or CMA-ES generations/iterations. Iterations are converted via "
            "evals = iters * (pop_size + 1) using the default CMA-ES pop_size."
        ),
    )
    parser.add_argument(
        "--handoff-timings",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Warmup timings to sweep over. Interpreted in the unit given by "
            "--timing-unit. Defaults: [250, 500, 1000, 1500, 2500] evaluations "
            "or [20, 50, 100, 150, 250] iterations."
        ),
    )
    parser.add_argument("--num-seeds", type=int, default=25)
    parser.add_argument("--memory-size", type=int, default=10)
    parser.add_argument("--sigma-rastrigin", type=float, default=2.0)
    parser.add_argument("--sigma-griewank", type=float, default=200.0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Process count for parallel run execution (>1 uses joblib loky backend).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/handoff/timing_sweep_multimodal"),
    )
    args = parser.parse_args()

    if args.handoff_timings is None:
        args.handoff_timings = (
            [250, 500, 1000, 1500, 2500]
            if args.timing_unit == "evaluations"
            else [20, 50, 100, 150, 250]
        )

    # Validate: every timing must convert to a budget strictly less than total.
    evals_per_gen = cmaes_evals_per_generation(args.dimensions)
    for raw in args.handoff_timings:
        evals = raw if args.timing_unit == "evaluations" else raw * evals_per_gen
        if evals >= args.total_budget:
            raise ValueError(
                f"Timing {raw} {args.timing_unit} -> {evals} evals exceeds "
                f"total-budget {args.total_budget}."
            )

    run_handoff_timing_sweep(
        dimensions=args.dimensions,
        total_budget=args.total_budget,
        timings=sorted(args.handoff_timings),
        timing_unit=args.timing_unit,
        num_seeds=args.num_seeds,
        initial_sigma_rastrigin=args.sigma_rastrigin,
        initial_sigma_griewank=args.sigma_griewank,
        memory_size=args.memory_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
