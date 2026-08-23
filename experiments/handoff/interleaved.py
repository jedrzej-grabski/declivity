"""Interleaved CMA-ES <-> L-BFGS-B handoff.

An extension of the one-shot ``CMAESLBFGSBHandoff``: instead of cutting over
to L-BFGS-B once, this alternates between the two for the whole budget.
Every ``cmaes_interval`` CMA-ES generations a short L-BFGS-B *side-probe*
fires from the current CMA-ES mean (with ``B_0 = C^{-1}`` — covariance only,
exactly as in the one-shot handoff), refines the running best until it
"stops advancing rapidly", and hands control straight back to CMA-ES, which
continues *untouched* (the probe does not feed back into the distribution).

Tracking the OVERALL BEST across both algorithms produces the characteristic
staircase: a gently descending CMA-ES backbone with sharp L-BFGS-B drops,
each deeper than the last as CMA-ES's covariance turns into an ever-better
Hessian model.

Test problem (default): a rotated, ill-conditioned Ellipsoid whose minimum
has been translated to *near a corner of the feasible box* with the new
``ShiftedFunction`` wrapper — so the bound constraints actually bite and
L-BFGS-B's projected-gradient / Cauchy-point machinery has work to do. This
mirrors the classic "optimum in the corner of the feasible region" benchmark.

Outputs (under ``plots/handoff/interleaved/<problem>/``):

- ``staircase.png`` — single-run dissection: overall-best staircase vs
  CMA-ES backbone vs L-BFGS-B bursts (the headline figure).
- ``convergence.png`` — multi-seed median + IQR for CMA-ES, L-BFGS-B, the
  one-shot handoff, and the interleaved scheme.
- ``final_fitness.png`` — final-fitness boxplot across seeds.

Run::

    PYTHONPATH=. uv run python experiments/handoff/interleaved.py
    PYTHONPATH=. uv run python experiments/handoff/interleaved.py --problem rastrigin
    PYTHONPATH=. uv run python experiments/handoff/interleaved.py --dimensions 20 --num-seeds 25
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.lbfgsb import ArmijoBacktracking
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.algorithms.lbfgsb.line_search import LineSearchStrategy
from declivity.benchmarking import (
    Benchmark,
    CMAESLBFGSBHandoff,
    InterleavedCMAESLocal,
    Problem,
    SingleAlgorithm,
)
from declivity.plotting import (
    plot_benchmark_boxplot,
    plot_benchmark_convergence,
    plot_interleaved_convergence,
)
from declivity.utils.benchmark_functions import (
    Rastrigin,
    RotatedEllipsoid,
    ShiftedFunction,
)
from declivity.utils.stopping_conditions import MaxEvaluations

plt.ioff()
plt.switch_backend("Agg")


COLORS = {
    "CMA-ES": "#c0392b",
    "L-BFGS-B": "#2980b9",
    "CMA-ES -> L-BFGS-B (one-shot)": "#8e44ad",
    "Interleaved CMA-ES + L-BFGS-B": "#e67e22",
}


def build_problem(
    problem_name: str,
    dimensions: int,
    corner_fraction: float,
    rotation: str,
) -> tuple[Problem, LineSearchStrategy | None]:
    """Build a near-corner shifted problem and the matching L-BFGS-B line search.

    Returns ``(problem, line_search)``; ``line_search`` is ``None`` for the
    smooth Ellipsoid (More-Thuente is fine) and Armijo for the rippled
    multimodal Rastrigin (More-Thuente bails on the cosine ripples).
    """
    if problem_name == "ellipsoid":
        base = RotatedEllipsoid(dimensions, rotation=rotation, seed=0)
        func = ShiftedFunction.near_corner(base, fraction=corner_fraction)
        name = f"ShiftedRotEllipsoid({rotation})"
        return Problem.from_benchmark(name, func), None

    if problem_name == "rastrigin":
        base = Rastrigin(dimensions)
        func = ShiftedFunction.near_corner(base, fraction=corner_fraction)
        return Problem.from_benchmark("ShiftedRastrigin", func), ArmijoBacktracking()

    raise ValueError(
        f"Unknown problem {problem_name!r}; use 'ellipsoid' or 'rastrigin'."
    )


def build_algorithms(
    total_budget: int,
    cmaes_warmup_budget: int,
    cmaes_interval: int,
    memory_size: int,
    probe_max_evals: int,
    probe_factr: float,
    line_search: LineSearchStrategy | None,
) -> list:
    """The four algorithms, all sharing one total evaluation budget."""

    def cmaes_config(dimensions: int) -> CMAESConfig:
        return CMAESConfig(dimensions=dimensions)

    def cmaes_warmup_config(dimensions: int) -> CMAESConfig:
        return CMAESConfig(dimensions=dimensions)

    def lbfgsb_config(dimensions: int) -> LBFGSBConfig:
        return LBFGSBConfig(
            dimensions=dimensions,
            m=memory_size,
            pgtol=1e-10,
            factr=0,
        )

    cmaes_only = SingleAlgorithm(
        name="CMA-ES",
        color=COLORS["CMA-ES"],
        algorithm=AlgorithmChoice.CMAES,
        config_factory=cmaes_config,
        stopping_condition=MaxEvaluations(total_budget),
    )

    lbfgsb_only = SingleAlgorithm(
        name="L-BFGS-B",
        color=COLORS["L-BFGS-B"],
        algorithm=AlgorithmChoice.LBFGSB,
        config_factory=lbfgsb_config,
        line_search=line_search,
        stopping_condition=MaxEvaluations(total_budget),
    )

    one_shot = CMAESLBFGSBHandoff(
        name="CMA-ES -> L-BFGS-B (one-shot)",
        color=COLORS["CMA-ES -> L-BFGS-B (one-shot)"],
        cmaes_config_factory=cmaes_warmup_config,
        lbfgsb_config_factory=lambda d: LBFGSBConfig(
            dimensions=d,
            m=memory_size,
            pgtol=1e-10,
            factr=0,
        ),
        transform="inverse",
        lbfgsb_line_search=line_search,
        cmaes_stopping_condition=MaxEvaluations(cmaes_warmup_budget),
        lbfgsb_stopping_condition=MaxEvaluations(total_budget - cmaes_warmup_budget),
    )

    interleaved = InterleavedCMAESLocal(
        name="Interleaved CMA-ES + L-BFGS-B",
        color=COLORS["Interleaved CMA-ES + L-BFGS-B"],
        cmaes_config_factory=cmaes_config,
        local_config_factory=lbfgsb_config,
        local_algorithm=AlgorithmChoice.LBFGSB,
        cmaes_interval=cmaes_interval,
        total_budget=total_budget,
        transform="inverse",
        probe_factr=probe_factr,
        probe_max_evals=probe_max_evals,
        local_line_search=line_search,
    )

    return [cmaes_only, lbfgsb_only, one_shot, interleaved]


def run_experiment(
    problem_name: str,
    dimensions: int,
    total_budget: int,
    cmaes_warmup_budget: int,
    cmaes_interval: int,
    num_seeds: int,
    memory_size: int,
    probe_max_evals: int,
    probe_factr: float,
    corner_fraction: float,
    rotation: str,
    headline_seed: int,
    num_workers: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    problem, line_search = build_problem(
        problem_name, dimensions, corner_fraction, rotation
    )
    algorithms = build_algorithms(
        total_budget,
        cmaes_warmup_budget,
        cmaes_interval,
        memory_size,
        probe_max_evals,
        probe_factr,
        line_search,
    )
    cmaes_only, _lbfgsb_only, _one_shot, interleaved = algorithms

    optimum = problem.function.global_minimum[0]
    print(
        f"\nProblem: {problem.name} (d={dimensions}, box=[{problem.lower_bound}, "
        f"{problem.upper_bound}]); optimum at corner-fraction {corner_fraction} "
        f"(x*[0]={optimum[0]:.2f}, f(x*)={float(problem.function(optimum)):.2e})"
    )

    # ---- Headline single-run staircase --------------------------------
    x0 = problem.starting_point(headline_seed)
    detail = interleaved.run_with_detail(problem, x0, headline_seed)
    cmaes_baseline = cmaes_only.run(problem, x0, headline_seed)
    print(
        f"Headline (seed={headline_seed}): {detail.num_bursts} bursts over "
        f"{detail.cmaes_generations} CMA-ES gens; interleaved final "
        f"f={detail.trace.final_fitness:.3e} vs CMA-ES-alone "
        f"f={cmaes_baseline.final_fitness:.3e}"
    )
    plot_interleaved_convergence(
        detail,
        baseline_trace=cmaes_baseline,
        title=(
            f"Interleaved CMA-ES <-> L-BFGS-B on {problem.name} "
            f"(d={dimensions}, seed={headline_seed})"
        ),
        save_path=output_dir / "staircase.png",
    )

    # ---- Multi-seed comparison ----------------------------------------
    bench = Benchmark(
        problems=[problem],
        algorithms=algorithms,
        seeds=list(range(num_seeds)),
        output_dir=output_dir,
        num_workers=num_workers,
    )
    bench.run(verbose=True)
    bench.print_summary()

    plot_benchmark_convergence(
        bench.traces,
        problems=[problem],
        algorithms=algorithms,
        title=(
            f"Interleaved vs one-shot vs standalone on {problem.name} "
            f"({dimensions}D, {num_seeds} seeds, budget {total_budget})"
        ),
        save_path=output_dir / "convergence.png",
    )
    plot_benchmark_boxplot(
        bench.traces,
        problems=[problem],
        algorithms=algorithms,
        title=f"Final fitness distribution ({dimensions}D, {num_seeds} seeds)",
        save_path=output_dir / "final_fitness.png",
    )
    print(f"\nPlots saved to: {output_dir.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", choices=["ellipsoid", "rastrigin"], default="ellipsoid"
    )
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--total-budget", type=int, default=8000)
    parser.add_argument(
        "--cmaes-warmup-budget",
        type=int,
        default=2000,
        help="Warm-up budget for the one-shot handoff baseline only.",
    )
    parser.add_argument(
        "--cmaes-interval",
        type=int,
        default=20,
        help="CMA-ES generations between consecutive L-BFGS-B probes (the handoff interval N).",
    )
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--memory-size", type=int, default=10)
    parser.add_argument(
        "--probe-max-evals",
        type=int,
        default=80,
        help="Hard cap on evaluations per L-BFGS-B probe. Short, frequent "
        "bursts (small cap) track CMA-ES's improving covariance and give "
        "the cleanest staircase; a large cap lets one early burst stall "
        "on a stale C^-1.",
    )
    parser.add_argument(
        "--probe-factr",
        type=float,
        default=1e7,
        help="Relative-decrease stop for each probe (larger -> bails sooner).",
    )
    parser.add_argument(
        "--corner-fraction",
        type=float,
        default=0.9,
        help="How far the optimum sits from the box centre toward a corner "
        "(0 = centre, 1 = exactly on the corner).",
    )
    parser.add_argument(
        "--rotation",
        choices=["random", "uniform_45", "golden"],
        default="random",
        help="Rotation applied to the Ellipsoid (ignored for Rastrigin).",
    )
    parser.add_argument("--headline-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("plots/handoff/interleaved") / args.problem

    run_experiment(
        problem_name=args.problem,
        dimensions=args.dimensions,
        total_budget=args.total_budget,
        cmaes_warmup_budget=args.cmaes_warmup_budget,
        cmaes_interval=args.cmaes_interval,
        num_seeds=args.num_seeds,
        memory_size=args.memory_size,
        probe_max_evals=args.probe_max_evals,
        probe_factr=args.probe_factr,
        corner_fraction=args.corner_fraction,
        rotation=args.rotation,
        headline_seed=args.headline_seed,
        num_workers=args.num_workers,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
