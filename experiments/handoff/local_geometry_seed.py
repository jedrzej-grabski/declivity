"""Covariance-seeded local optimizers: CMA-ES -> {L-BFGS-B, Powell}.

The generalization of the CMA-ES -> L-BFGS-B covariance handoff to any
single-point local optimizer, through one shared ``InitialGeometry`` object and
the uniform ``CMAESLocalHandoff``. After a CMA-ES warm-up the learned covariance
seeds each local optimizer via its natural mechanism:

- **L-BFGS-B** — initial Hessian ``B_0 = C^{-1}``.
- **Powell**   — initial search-direction set = the eigenvectors of ``C``
  (un-rotates coordinate descent onto the landscape's principal axes).

For each optimizer we compare, on an anisotropic *rotated* (non-axis-aligned)
benchmark where directional curvature matters most:

1. CMA-ES standalone (shared reference).
2. the local optimizer standalone (from a random x0).
3. CMA-ES -> local, **covariance-seeded** (``transform="inverse"``).
4. CMA-ES -> local, **identity control** (``transform="identity"`` — same warm-up
   x0, no covariance information) — isolates "does the covariance shape help?"
   from "does sharing the CMA-ES warm-up point help?".

Same seed => same x0 and same CMA-ES RNG path, so curves (1), (3), (4) coincide
up to the handoff. One convergence plot + one final-fitness boxplot per optimizer.

Run::

    PYTHONPATH=. pdm run python experiments/handoff/local_geometry_seed.py
    PYTHONPATH=. pdm run python experiments/handoff/local_geometry_seed.py \
        --dimensions 20 --num-seeds 15 --rotation random
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.algorithms.powell.config import PowellConfig
from declivity.benchmarking import (
    Benchmark,
    CMAESLocalHandoff,
    Problem,
    SingleAlgorithm,
)
from declivity.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from declivity.utils.benchmark_functions import RotatedEllipsoid
from declivity.utils.stopping_conditions import MaxEvaluations


plt.ioff()
plt.switch_backend("Agg")


# One color scheme reused across optimizers (the four roles are the same).
COLORS = {
    "cmaes":      "#e74c3c",
    "standalone": "#95a5a6",
    "covariance": "#2ecc71",
    "identity":   "#9b59b6",
}

# (display label, factory choice, config factory) for each local optimizer.
LOCAL_SPECS = [
    ("L-BFGS-B", AlgorithmChoice.LBFGSB, lambda d: LBFGSBConfig(dimensions=d)),
    ("Powell",   AlgorithmChoice.POWELL, lambda d: PowellConfig(dimensions=d)),
]


def build_algorithms(
    label: str,
    local_algorithm: AlgorithmChoice,
    local_config_factory,
    initial_sigma: float,
    warmup_budget: int,
    total_budget: int,
):
    """The four runners compared for one local optimizer, sharing a budget.

    The handoffs split the budget: ``warmup_budget`` evaluations on CMA-ES, the
    remainder on the local optimizer.
    """
    refinement_budget = total_budget - warmup_budget

    def cmaes_cfg(d: int) -> CMAESConfig:
        return CMAESConfig(dimensions=d, sigma=initial_sigma)

    cmaes_only = SingleAlgorithm(
        name="CMA-ES",
        color=COLORS["cmaes"],
        algorithm=AlgorithmChoice.CMAES,
        config_factory=cmaes_cfg,
        stopping_condition=MaxEvaluations(total_budget),
    )

    local_only = SingleAlgorithm(
        name=f"{label}",
        color=COLORS["standalone"],
        algorithm=local_algorithm,
        config_factory=local_config_factory,
        stopping_condition=MaxEvaluations(total_budget),
    )

    handoff_covariance = CMAESLocalHandoff(
        name=f"CMA-ES -> {label} (covariance)",
        color=COLORS["covariance"],
        local_algorithm=local_algorithm,
        cmaes_config_factory=cmaes_cfg,
        local_config_factory=local_config_factory,
        transform="inverse",
        cmaes_stopping_condition=MaxEvaluations(warmup_budget),
        local_stopping_condition=MaxEvaluations(refinement_budget),
    )

    handoff_identity = CMAESLocalHandoff(
        name=f"CMA-ES -> {label} (identity)",
        color=COLORS["identity"],
        local_algorithm=local_algorithm,
        cmaes_config_factory=cmaes_cfg,
        local_config_factory=local_config_factory,
        transform="identity",
        cmaes_stopping_condition=MaxEvaluations(warmup_budget),
        local_stopping_condition=MaxEvaluations(refinement_budget),
    )

    return [cmaes_only, local_only, handoff_covariance, handoff_identity]


def run(
    dimensions: int,
    rotation: str,
    total_budget: int,
    warmup_budget: int,
    initial_sigma: float,
    num_seeds: int,
    num_workers: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(num_seeds))

    function = RotatedEllipsoid(dimensions, rotation=rotation, seed=0)
    problem = Problem.from_benchmark(
        f"RotatedEllipsoid-{dimensions}D-{rotation}", function
    )

    for label, local_algorithm, local_config_factory in LOCAL_SPECS:
        algorithms = build_algorithms(
            label, local_algorithm, local_config_factory,
            initial_sigma, warmup_budget, total_budget,
        )
        sub_dir = output_dir / label.lower().replace("-", "")
        bench = Benchmark(
            problems=[problem],
            algorithms=algorithms,
            seeds=seeds,
            output_dir=sub_dir,
            num_workers=num_workers,
        )
        bench.run(verbose=True)
        bench.print_summary()

        plot_benchmark_convergence(
            bench.traces,
            problems=[problem],
            algorithms=algorithms,
            title=(
                f"Covariance-seeded {label}: rotated ellipsoid "
                f"({dimensions}D {rotation}, {num_seeds} seeds, "
                f"warm-up {warmup_budget}/{total_budget})"
            ),
            save_path=sub_dir / "convergence.png",
        )
        plot_benchmark_boxplot(
            bench.traces,
            problems=[problem],
            algorithms=algorithms,
            title=f"Final fitness — {label} ({dimensions}D {rotation}, {num_seeds} seeds)",
            save_path=sub_dir / "final_fitness.png",
        )

    print(f"\nPlots saved under: {output_dir.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument(
        "--rotation", type=str, default="random",
        choices=["none", "uniform_45", "golden", "random"],
        help="Rotation applied to the ellipsoid (anisotropy is only "
        "'un-axis-aligned' for the rotated cases — see the rotation study).",
    )
    parser.add_argument("--total-budget", type=int, default=6000)
    parser.add_argument("--warmup-budget", type=int, default=2000)
    parser.add_argument("--initial-sigma", type=float, default=20.0)
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("plots/handoff/local_geometry_seed"),
    )
    args = parser.parse_args()

    run(
        dimensions=args.dimensions,
        rotation=args.rotation,
        total_budget=args.total_budget,
        warmup_budget=args.warmup_budget,
        initial_sigma=args.initial_sigma,
        num_seeds=args.num_seeds,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
