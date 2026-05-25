"""Anisotropic multimodal handoff study.

Built on top of RippledEllipsoid:

    f(x) = sum(scale_i * x_i**2 + amplitude * (1 - cos(2 pi x_i)))

When the condition number is large the local Hessian at the global
optimum spans many orders of magnitude; when the rotation is non-trivial
that anisotropy is no longer axis-aligned. This is the regime where
L-BFGS-B with B_0 = I is forced to spend many iterations learning the
curvature, and where the C^{-1} handoff from CMA-ES should pull
decisively ahead.

The script sweeps a configurable grid of (dimension, memory size,
condition number) and produces a combined convergence + boxplot pair.
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
from src.utils.benchmark_functions import RippledEllipsoid, RotatedFunction


plt.ioff()
plt.switch_backend("Agg")


COLORS = {
    "CMA-ES":             "#e74c3c",
    "L-BFGS-B":           "#3498db",
    "Handoff (C^-1)":     "#2ecc71",
    "Handoff (identity)": "#9b59b6",
}


def build_function(
    dimensions: int,
    condition: float,
    rotated: bool,
    rotation_seed: int,
    amplitude: float = 10.0,
):
    base = RippledEllipsoid(
        dimensions, condition=condition, amplitude=amplitude, bound=5.12
    )
    if not rotated:
        return base
    return RotatedFunction(base, rotation="random", seed=rotation_seed)


def build_algorithms(
    total_budget: int,
    cmaes_warmup_budget: int,
    initial_sigma: float,
    memory_size: int,
    pgtol: float = 1e-10,
    factr: float = 0.0,
):
    lbfgsb_handoff_budget = total_budget - cmaes_warmup_budget

    common_lbfgsb_kwargs = dict(
        m=memory_size, pgtol=pgtol, factr=factr,
        line_search=LineSearchMethod.ARMIJO,
    )

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
            dimensions=d, budget=total_budget, **common_lbfgsb_kwargs,
        ),
    )

    handoff_inverse = CMAESLBFGSBHandoff(
        name="Handoff (C^-1)",
        color=COLORS["Handoff (C^-1)"],
        cmaes_config_factory=lambda d: CMAESConfig(
            dimensions=d, budget=cmaes_warmup_budget, sigma=initial_sigma,
        ),
        lbfgsb_config_factory=lambda d: LBFGSBConfig(
            dimensions=d, budget=lbfgsb_handoff_budget, **common_lbfgsb_kwargs,
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
            dimensions=d, budget=lbfgsb_handoff_budget, **common_lbfgsb_kwargs,
        ),
        transform="identity",
    )

    return [cmaes_only, lbfgsb_only, handoff_inverse, handoff_identity]


def run_one_panel(
    dimensions: int,
    condition: float,
    memory_size: int,
    rotated: bool,
    total_budget: int,
    cmaes_warmup_budget: int,
    num_seeds: int,
    rotation_seed: int,
    num_workers: int,
    output_dir: Path,
    amplitude: float = 10.0,
    pgtol: float = 1e-10,
    factr: float = 0.0,
    initial_sigma: float = 2.0,
) -> tuple[Problem, list, dict]:
    function = build_function(
        dimensions, condition, rotated, rotation_seed, amplitude=amplitude
    )
    rotated_tag = "rot" if rotated else "axis"
    amp_tag = f"a{amplitude:g}"
    panel_name = (
        f"Rippled-c{int(condition)}-{amp_tag}-{rotated_tag}"
        f"-d{dimensions}-m{memory_size}"
    )
    problem = Problem.from_benchmark(panel_name, function)

    # Initial sigma should scale with the bound. RippledEllipsoid has bound 5.12.
    algorithms = build_algorithms(
        total_budget=total_budget,
        cmaes_warmup_budget=cmaes_warmup_budget,
        initial_sigma=initial_sigma,
        memory_size=memory_size,
        pgtol=pgtol,
        factr=factr,
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


def render_combined(panels, output_dir: Path, title: str) -> None:
    problems = [p for p, _, _ in panels]
    representative_algorithms = panels[0][1]
    combined_traces: dict = {}
    for _, _, traces in panels:
        for key, value in traces.items():
            combined_traces[key] = value

    plotter = BenchmarkPlotter(
        problems=problems,
        algorithms=representative_algorithms,
        traces=combined_traces,
        output_dir=output_dir,
    )
    plotter.plot_convergence_grid(
        save_path=output_dir / "convergence.png",
        title=title,
    )
    plotter.plot_final_fitness_boxplot(
        save_path=output_dir / "final_fitness.png",
        title=f"Final fitness distribution ({title})",
    )
    print(f"\nCombined plot saved to {output_dir.absolute()}/convergence.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--conditions", type=float, nargs="+", default=[1000.0])
    parser.add_argument("--memory", type=int, nargs="+", default=[5])
    parser.add_argument(
        "--amplitude", type=float, default=10.0,
        help="Ripple amplitude. Lower = less multimodal but better conditioned for L-BFGS-B; default 10 (Rastrigin-like).",
    )
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
        "--pgtol", type=float, default=1e-10,
        help="L-BFGS-B projected gradient tolerance (looser = earlier stop).",
    )
    parser.add_argument(
        "--factr", type=float, default=0.0,
        help="L-BFGS-B f-value relative tolerance factor (0 disables).",
    )
    parser.add_argument(
        "--initial-sigma", type=float, default=2.0,
        help="Initial sigma for CMA-ES.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("plots/hybrid/rippled_ellipsoid_handoff"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels: list = []
    for d in args.dimensions:
        for cond in args.conditions:
            for m in args.memory:
                panels.append(
                    run_one_panel(
                        dimensions=d,
                        condition=cond,
                        memory_size=m,
                        rotated=args.rotated,
                        total_budget=args.total_budget,
                        cmaes_warmup_budget=args.cmaes_warmup_budget,
                        num_seeds=args.num_seeds,
                        rotation_seed=args.rotation_seed,
                        num_workers=args.num_workers,
                        output_dir=args.output_dir,
                        amplitude=args.amplitude,
                        pgtol=args.pgtol,
                        factr=args.factr,
                        initial_sigma=args.initial_sigma,
                    )
                )

    rot_str = "rotated" if args.rotated else "axis-aligned"
    title = (
        f"Rippled Ellipsoid ({rot_str}): C^-1 vs identity handoff "
        f"({args.num_seeds} seeds, budget {args.total_budget}, warmup {args.cmaes_warmup_budget})"
    )
    render_combined(panels, args.output_dir, title)


if __name__ == "__main__":
    main()
