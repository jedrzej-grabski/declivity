"""Does the CMA-ES covariance actually help when L-BFGS-B takes over?

In the unrotated Rastrigin/Griewank run, the C^{-1} handoff and the
identity handoff converged to the same final fitness in the same number
of evaluations. The reason: at every Rastrigin local minimum the
Hessian is *exactly isotropic* (eigenvalues all 2 + 40 pi^2), so there
is no curvature direction for CMA-ES to learn. Griewank does have
anisotropic curvature (cond ~ d), but in the coordinate-aligned form
identity + a few BFGS corrections recovers it just as quickly.

This script tests the case where the covariance *must* matter:

  - Rotate Griewank with a random orthogonal matrix. The eigenvalues of
    the local Hessian stay the same (cond ~ d) but they're no longer
    axis-aligned. Identity B_0 starts blind to the rotation.

  - Push dimension up so that the L-BFGS-B memory size m is small
    relative to d: corrections can span at most m directions, so the
    remaining d - m have to come from B_0.

  - Vary handoff timing to find the point at which CMA-ES has learned
    enough of the rotation to be useful.

Sweeps are configurable from the CLI. The default sweep produces the
plot grid for the supervisor report.
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
from src.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from src.utils.benchmark_functions import Griewank, Rastrigin, RotatedFunction


plt.ioff()
plt.switch_backend("Agg")


COLORS = {
    "CMA-ES":                          "#e74c3c",
    "L-BFGS-B":                        "#3498db",
    "Handoff (C^-1)":                  "#2ecc71",
    "Handoff (identity)":              "#9b59b6",
}


def make_rotated(base_name: str, dimensions: int, rotation_seed: int):
    """Return a RotatedFunction wrapping the named base function."""
    base_cls = {"Griewank": Griewank, "Rastrigin": Rastrigin}[base_name]
    base = base_cls(dimensions)
    return RotatedFunction(base, rotation="random", seed=rotation_seed)


def build_algorithms(
    total_budget: int,
    cmaes_warmup_budget: int,
    initial_sigma: float,
    memory_size: int,
):
    """Four algorithms: CMA-ES, L-BFGS-B, two handoffs (C^-1 vs identity)."""
    lbfgsb_handoff_budget = total_budget - cmaes_warmup_budget

    cmaes_only = SingleAlgorithm(
        name="CMA-ES",
        color=COLORS["CMA-ES"],
        algorithm=AlgorithmChoice.CMAES,
        config_factory=lambda d: CMAESConfig(
            dimensions=d, budget=total_budget, sigma=initial_sigma,
        ),
    )

    lbfgsb_only = SingleAlgorithm(
        name="L-BFGS-B",
        color=COLORS["L-BFGS-B"],
        algorithm=AlgorithmChoice.LBFGSB,
        config_factory=lambda d: LBFGSBConfig(
            dimensions=d, budget=total_budget, m=memory_size,
            pgtol=1e-10, factr=0,
            line_search=LineSearchMethod.ARMIJO,
        ),
    )

    handoff_inverse = CMAESLBFGSBHandoff(
        name="Handoff (C^-1)",
        color=COLORS["Handoff (C^-1)"],
        cmaes_config_factory=lambda d: CMAESConfig(
            dimensions=d, budget=cmaes_warmup_budget, sigma=initial_sigma,
        ),
        lbfgsb_config_factory=lambda d: LBFGSBConfig(
            dimensions=d, budget=lbfgsb_handoff_budget, m=memory_size,
            pgtol=1e-10, factr=0,
            line_search=LineSearchMethod.ARMIJO,
        ),
        transform="inverse",
    )

    handoff_identity = CMAESLBFGSBHandoff(
        name="Handoff (identity)",
        color=COLORS["Handoff (identity)"],
        cmaes_config_factory=lambda d: CMAESConfig(
            dimensions=d, budget=cmaes_warmup_budget, sigma=initial_sigma,
        ),
        lbfgsb_config_factory=lambda d: LBFGSBConfig(
            dimensions=d, budget=lbfgsb_handoff_budget, m=memory_size,
            pgtol=1e-10, factr=0,
            line_search=LineSearchMethod.ARMIJO,
        ),
        transform="identity",
    )

    return [cmaes_only, lbfgsb_only, handoff_inverse, handoff_identity]


def sigma_for(base_name: str) -> float:
    """Sensible initial sigma per problem (matches multimodal_handoff_benchmark)."""
    if base_name == "Griewank":
        return 200.0
    if base_name == "Rastrigin":
        return 2.0
    raise ValueError(base_name)


def run_one_panel(
    base_name: str,
    dimensions: int,
    rotated: bool,
    memory_size: int,
    total_budget: int,
    cmaes_warmup_budget: int,
    num_seeds: int,
    rotation_seed: int,
    num_workers: int,
    output_dir: Path,
) -> tuple[Problem, list, dict]:
    """One panel = one (function, d, rotated, m) configuration."""
    function = (
        make_rotated(base_name, dimensions, rotation_seed)
        if rotated
        else {"Griewank": Griewank, "Rastrigin": Rastrigin}[base_name](dimensions)
    )
    rotated_tag = "rot" if rotated else "axis"
    panel_name = f"{base_name}-{rotated_tag}-d{dimensions}-m{memory_size}"
    problem = Problem.from_benchmark(panel_name, function)

    algorithms = build_algorithms(
        total_budget=total_budget,
        cmaes_warmup_budget=cmaes_warmup_budget,
        initial_sigma=sigma_for(base_name),
        memory_size=memory_size,
    )

    bench = Benchmark(
        problems=[problem],
        algorithms=algorithms,
        seeds=list(range(num_seeds)),
        output_dir=output_dir / panel_name,
        num_workers=num_workers,
    )
    bench.run(verbose=True)
    bench.print_summary()
    return problem, algorithms, bench.traces


def render_combined(
    panels: list[tuple[Problem, list, dict]],
    output_dir: Path,
    title: str,
    cmaes_warmup_budget: int,
) -> None:
    """Render all panels into one combined plot grid."""
    problems = [p for p, _, _ in panels]
    representative_algorithms = panels[0][1]
    combined_traces: dict = {}
    for _, _, traces in panels:
        for key, value in traces.items():
            combined_traces[key] = value

    plot_benchmark_convergence(
        combined_traces,
        problems=problems,
        algorithms=representative_algorithms,
        title=title,
        save_path=output_dir / "convergence.png",
    )
    plot_benchmark_boxplot(
        combined_traces,
        problems=problems,
        algorithms=representative_algorithms,
        title=f"Final fitness distribution ({title})",
        save_path=output_dir / "final_fitness.png",
    )

    print(f"\nCombined plot saved to {output_dir.absolute()}/convergence.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bases", nargs="+", default=["Griewank"],
        help="Base function name(s): Griewank or Rastrigin (or both).",
    )
    parser.add_argument("--dimensions", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--memory", type=int, nargs="+", default=[10])
    parser.add_argument(
        "--rotated", action="store_true", default=True,
        help="Use rotated functions (default true; pass --no-rotated to disable).",
    )
    parser.add_argument("--no-rotated", dest="rotated", action="store_false")
    parser.add_argument("--total-budget", type=int, default=10000)
    parser.add_argument("--cmaes-warmup-budget", type=int, default=2500)
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--rotation-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("plots/handoff/multimodal_rotated"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[Problem, list, dict]] = []
    for base_name in args.bases:
        for d in args.dimensions:
            for m in args.memory:
                panels.append(
                    run_one_panel(
                        base_name=base_name,
                        dimensions=d,
                        rotated=args.rotated,
                        memory_size=m,
                        total_budget=args.total_budget,
                        cmaes_warmup_budget=args.cmaes_warmup_budget,
                        num_seeds=args.num_seeds,
                        rotation_seed=args.rotation_seed,
                        num_workers=args.num_workers,
                        output_dir=args.output_dir,
                    )
                )

    title = (
        f"Rotated multimodal handoff: C^-1 vs identity "
        f"({args.num_seeds} seeds, budget {args.total_budget}, warmup {args.cmaes_warmup_budget})"
    )
    render_combined(panels, args.output_dir, title, args.cmaes_warmup_budget)


if __name__ == "__main__":
    main()
