"""Handoff timing sweep — one panel per experimental family.

Builds the same problem under each named experimental family and runs a
sweep of warmup budgets *expressed in CMA-ES generations* (translated
to evaluations using the algorithm's default population size).

Families supported (via ``--family``):
- ``baseline``       — Rastrigin and Griewank, d=10, m=10. The original puzzle.
- ``reproduce_old``  — Pure rotated Ellipsoid, d=50, m=5. Reproduces the old finding.
- ``low_amp``        — Rotated RippledEllipsoid, amp=0.1, d=30, m=5. C^-1 wins dramatically.
- ``multimodal``     — Rotated RippledEllipsoid, amp=1, d=50, m=5. Genuinely multimodal.

Each family has its own sensible defaults but the script also takes
overrides for total budget, warmup generations, num seeds, num workers,
etc.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.lbfgsb.config import LBFGSBConfig, LineSearchMethod
from src.benchmarking import (
    Benchmark,
    CMAESLBFGSBHandoff,
    Problem,
    SingleAlgorithm,
)
from src.plotting import plot_benchmark_convergence
from src.utils.benchmark_functions import (
    Griewank,
    Rastrigin,
    RippledEllipsoid,
    RotatedFunction,
)


plt.ioff()
plt.switch_backend("Agg")


PALETTE_INVERSE = ["#a8e6c9", "#74c69d", "#40916c", "#2d6a4f", "#1b4332"]
PALETTE_IDENTITY = ["#e9d8fd", "#c5a3f5", "#9b59b6", "#7e3aa6", "#5b2785"]

REFERENCE_COLORS = {
    "CMA-ES":   "#e74c3c",
    "L-BFGS-B": "#3498db",
}


def cmaes_evals_per_generation(dimensions: int) -> int:
    """CMA-ES costs pop_size + 1 evaluations per generation."""
    return CMAESConfig(dimensions=dimensions).population_size + 1


def gens_to_evals(gens: int, dimensions: int) -> int:
    return gens * cmaes_evals_per_generation(dimensions)


# ---------- family definitions ----------

def build_family(family: str, rotation_seed: int = 42):
    """Return (problems, dimensions, sigma, total_budget, memory, ls_settings).

    ``ls_settings`` is a dict of L-BFGS-B kwargs to pass through.
    """
    if family == "baseline":
        return {
            "problems": [
                ("Rastrigin", Rastrigin(10), 2.0),
                ("Griewank",  Griewank(10),  200.0),
            ],
            "dimensions": 10,
            "total_budget": 6000,
            "memory_size": 10,
            "ls_settings": dict(pgtol=1e-10, factr=0, line_search=LineSearchMethod.ARMIJO),
        }
    if family == "reproduce_old":
        base = RippledEllipsoid(50, condition=1e6, amplitude=0.0)
        return {
            "problems": [
                ("Ellipsoid-rot-d50",
                 RotatedFunction(base, rotation="random", seed=rotation_seed),
                 10.0),
            ],
            "dimensions": 50,
            "total_budget": 10000,
            "memory_size": 5,
            "ls_settings": dict(pgtol=1e-8, factr=1e7, line_search=LineSearchMethod.ARMIJO),
        }
    if family == "low_amp":
        base = RippledEllipsoid(30, condition=1e6, amplitude=0.1)
        return {
            "problems": [
                ("RippledEllipsoid-a0.1-rot-d30",
                 RotatedFunction(base, rotation="random", seed=rotation_seed),
                 2.0),
            ],
            "dimensions": 30,
            "total_budget": 10000,
            "memory_size": 5,
            "ls_settings": dict(pgtol=1e-10, factr=0, line_search=LineSearchMethod.ARMIJO),
        }
    if family == "multimodal":
        base = RippledEllipsoid(50, condition=1e6, amplitude=1.0)
        return {
            "problems": [
                ("RippledEllipsoid-a1-rot-d50",
                 RotatedFunction(base, rotation="random", seed=rotation_seed),
                 2.0),
            ],
            "dimensions": 50,
            "total_budget": 10000,
            "memory_size": 5,
            "ls_settings": dict(pgtol=1e-8, factr=1e7, line_search=LineSearchMethod.ARMIJO),
        }
    raise ValueError(f"unknown family {family!r}")


def default_warmup_gens(family: str) -> list[int]:
    """Generations to use as the timing-sweep axis per family."""
    return {
        "baseline":       [10, 30, 75, 150, 300],
        "reproduce_old":  [20, 75, 150, 300, 500],
        "low_amp":        [50, 100, 200, 350, 500],
        "multimodal":     [30, 75, 150, 250, 400],
    }[family]


# ---------- algorithm construction ----------

def build_algorithms(
    family_cfg: dict,
    warmup_gens_list: list[int],
    initial_sigma: float,
):
    dimensions = family_cfg["dimensions"]
    total_budget = family_cfg["total_budget"]
    memory_size = family_cfg["memory_size"]
    ls_settings = family_cfg["ls_settings"]

    lbfgsb_kwargs = dict(**ls_settings, m=memory_size)

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
    )

    handoffs: list[CMAESLBFGSBHandoff] = []
    for idx, warmup_gens in enumerate(warmup_gens_list):
        warmup_evals = gens_to_evals(warmup_gens, dimensions)
        post = total_budget - warmup_evals
        if post <= 0:
            raise ValueError(
                f"warmup {warmup_gens} gen ({warmup_evals} evals) >= total {total_budget}"
            )

        handoffs.append(CMAESLBFGSBHandoff(
            name=f"C^-1 @ {warmup_gens} gen",
            color=PALETTE_INVERSE[idx % len(PALETTE_INVERSE)],
            cmaes_config_factory=lambda d, b=warmup_evals: CMAESConfig(
                dimensions=d, budget=b, sigma=initial_sigma,
            ),
            lbfgsb_config_factory=lambda d, b=post: LBFGSBConfig(
                dimensions=d, budget=b, **lbfgsb_kwargs,
            ),
            transform="inverse",
        ))
        handoffs.append(CMAESLBFGSBHandoff(
            name=f"identity @ {warmup_gens} gen",
            color=PALETTE_IDENTITY[idx % len(PALETTE_IDENTITY)],
            cmaes_config_factory=lambda d, b=warmup_evals: CMAESConfig(
                dimensions=d, budget=b, sigma=initial_sigma,
            ),
            lbfgsb_config_factory=lambda d, b=post: LBFGSBConfig(
                dimensions=d, budget=b, **lbfgsb_kwargs,
            ),
            transform="identity",
        ))

    return [cmaes_only, lbfgsb_only, *handoffs]


def run(
    family: str,
    warmup_gens_list: list[int],
    num_seeds: int,
    num_workers: int,
    output_dir: Path,
    rotation_seed: int = 42,
) -> None:
    cfg = build_family(family, rotation_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems: list[Problem] = []
    for name, func, _sigma in cfg["problems"]:
        problems.append(Problem.from_benchmark(name, func))

    combined_traces: dict = {}
    representative_algorithms: list = []

    for (name, func, sigma), problem in zip(cfg["problems"], problems):
        algorithms = build_algorithms(cfg, warmup_gens_list, initial_sigma=sigma)

        bench = Benchmark(
            problems=[problem],
            algorithms=algorithms,
            seeds=list(range(num_seeds)),
            output_dir=output_dir / problem.name,
            num_workers=num_workers,
        )
        bench.run(verbose=True)
        bench.print_summary()

        for key, traces in bench.traces.items():
            combined_traces[key] = traces
        # Reuse the algorithm list metadata (name + color) for plotting.
        if not representative_algorithms:
            representative_algorithms = algorithms

    titles = {
        "baseline":      "Handoff timing — baseline (Rastrigin, Griewank d=10, m=10)",
        "reproduce_old": "Handoff timing — pure rotated Ellipsoid d=50, m=5",
        "low_amp":       "Handoff timing — rotated RippledEllipsoid amp=0.1 d=30 m=5",
        "multimodal":    "Handoff timing — rotated RippledEllipsoid amp=1 d=50 m=5",
    }
    plot_benchmark_convergence(
        combined_traces,
        problems=problems,
        algorithms=representative_algorithms,
        save_path=output_dir / "convergence.png",
        title=titles[family],
        show_iqr=False,
        figsize_per_panel=(9.5, 6.5),
        legend_fontsize=8,
        floor=1e-22 if family == "low_amp" else 1e-12,
    )

    print(f"\nWrote {output_dir}/convergence.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", required=True,
        choices=("baseline", "reproduce_old", "low_amp", "multimodal"),
    )
    parser.add_argument(
        "--warmup-gens", type=int, nargs="+", default=None,
        help="CMA-ES generations to use as warmup timings. Defaults to a sensible per-family list.",
    )
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--rotation-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.warmup_gens is None:
        args.warmup_gens = default_warmup_gens(args.family)

    if args.output_dir is None:
        args.output_dir = Path(f"plots/report/timing_{args.family}")

    run(
        family=args.family,
        warmup_gens_list=sorted(args.warmup_gens),
        num_seeds=args.num_seeds,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        rotation_seed=args.rotation_seed,
    )


if __name__ == "__main__":
    main()
