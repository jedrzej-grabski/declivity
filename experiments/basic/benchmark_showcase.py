"""Showcase: the benchmarking framework's extension hierarchy, one grid.

The companion to ``plotter_showcase.py``. Where that script shows the
*plotter* needs no per-algorithm code, this one shows the *runner harness*
doesn't either: a single :class:`Benchmark` grid runs four structurally
different runners side by side because every one of them satisfies the same
three-attribute contract — ``name``, ``color``, ``run(problem, x0, seed)``.

The four runners span every rung of the extension hierarchy:

1. ``SingleAlgorithm(CMA-ES)``    — concrete wrapper, one factory optimizer.
2. ``SingleAlgorithm(L-BFGS-B)``  — same wrapper, a different algorithm
   *family* (quasi-Newton vs. evolutionary). The grid can't tell.
3. ``CMAESLBFGSBHandoff``         — the pre-built two-phase ``HandoffAlgorithm``.
4. ``MultiStartCMAES``            — a custom ``BenchmarkAlgorithm`` defined
   right here in this file (restarts don't fit ``SingleAlgorithm`` or the
   two-phase ``HandoffAlgorithm``), proving the harness is open at the bottom.

``Benchmark`` runs all 2 problems x 4 runners x N seeds, persists lean
``RunTrace``s to ``traces.json`` (so a re-plot never re-runs an optimizer),
and the headline figure is the **multi-seed convergence** — a median
best-fitness line with a 25/75 IQR band per runner, one panel per problem.
Every runner's curve is read off ``bench.traces`` with no per-runner code:
the handoff's vertical marker and the custom multi-start's restart steps are
all drawn from the same trace dict by the same single call.

Output: ``plots/basic/benchmark_showcase/``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.benchmarking import (
    Benchmark,
    BenchmarkAlgorithm,
    CMAESLBFGSBHandoff,
    Problem,
    RunTrace,
    SingleAlgorithm,
    load_traces_json,
)
from declivity.core.algorithm_factory import AlgorithmFactory
from declivity.plotting import plot_benchmark_convergence
from declivity.utils.benchmark_functions import Rastrigin, Rosenbrock


plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/basic/benchmark_showcase")
DIMENSIONS = 10
NUM_SEEDS = 15
TOTAL_BUDGET = 4000

# One palette, carried on each runner's `.color`. The plotter reads it off
# the AlgorithmRun — same hue per algorithm across every figure.
COLORS = {
    "CMA-ES":             "#e74c3c",
    "L-BFGS-B":           "#3498db",
    "CMA-ES -> L-BFGS-B": "#2ecc71",
    "Multi-start CMA-ES": "#9b59b6",
}


# ---------------------------------------------------------------------------
# The custom runner — the most general extension point.
#
# Multi-start CMA-ES fits neither SingleAlgorithm (it makes several optimizer
# calls) nor HandoffAlgorithm (it is not warmup -> refinement, just N
# independent restarts with the best kept). So it subclasses
# BenchmarkAlgorithm and implements run() directly — and then drops into the
# same algorithm list as everything else.
# ---------------------------------------------------------------------------

@dataclass
class MultiStartCMAES(BenchmarkAlgorithm):
    """CMA-ES restarted ``num_restarts`` times, each from a fresh x0.

    Restart 0 starts from the framework-provided ``x0`` (so seed 0 is a fair
    head-to-head with plain CMA-ES); later restarts draw their own start from
    a per-restart sub-seed, keeping the whole run deterministic given
    ``seed``. The trace is the running best across all restarts.
    """

    name: str
    color: str
    config_factory: Callable[[int], CMAESConfig]
    num_restarts: int = 4

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        rng = np.random.default_rng(seed)
        restart_seeds = rng.integers(0, 2**31, size=self.num_restarts)

        evaluations: list[int] = []
        best_fitness: list[float] = []
        running_evals = 0
        running_best = float("inf")

        for restart_index, restart_seed in enumerate(restart_seeds):
            if restart_index == 0:
                restart_x0 = x0
            else:
                restart_rng = np.random.default_rng(int(restart_seed))
                restart_x0 = restart_rng.uniform(
                    problem.lower_bound, problem.upper_bound, size=problem.dimensions
                )

            result = AlgorithmFactory.create_optimizer(
                AlgorithmChoice.CMAES,
                problem.function,
                restart_x0,
                self.config_factory(problem.dimensions),
                lower_bounds=problem.lower_bound,
                upper_bounds=problem.upper_bound,
                seed=int(restart_seed),
            ).optimize()

            for evaluation, fitness in zip(
                result.diagnostic.evaluations, result.diagnostic.best_fitness
            ):
                running_best = min(running_best, float(fitness))
                evaluations.append(int(evaluation) + running_evals)
                best_fitness.append(running_best)

            running_evals += result.evaluations
            running_best = min(running_best, float(result.best_fitness))

        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=evaluations,
            best_fitness=best_fitness,
            final_evaluations=running_evals,
            final_fitness=running_best,
        )


def build_algorithms() -> list:
    """The four heterogeneous runners — one per rung of the hierarchy."""
    restarts = 4
    return [
        # Rung 1 & 2: concrete SingleAlgorithm, two different families.
        SingleAlgorithm(
            name="CMA-ES",
            color=COLORS["CMA-ES"],
            algorithm=AlgorithmChoice.CMAES,
            config_factory=lambda d: CMAESConfig(dimensions=d, budget=TOTAL_BUDGET),
        ),
        SingleAlgorithm(
            name="L-BFGS-B",
            color=COLORS["L-BFGS-B"],
            algorithm=AlgorithmChoice.LBFGSB,
            config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=TOTAL_BUDGET),
        ),
        # Rung 3: the pre-built two-phase HandoffAlgorithm.
        CMAESLBFGSBHandoff(
            name="CMA-ES -> L-BFGS-B",
            color=COLORS["CMA-ES -> L-BFGS-B"],
            cmaes_config_factory=lambda d: CMAESConfig(dimensions=d, budget=1600),
            lbfgsb_config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=2400),
            transform="inverse",
        ),
        # Rung 4: a custom BenchmarkAlgorithm from this very file.
        MultiStartCMAES(
            name="Multi-start CMA-ES",
            color=COLORS["Multi-start CMA-ES"],
            config_factory=lambda d: CMAESConfig(
                dimensions=d, budget=TOTAL_BUDGET // restarts
            ),
            num_restarts=restarts,
        ),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Two problem characters so the convergence curves have a story to tell: a
    # multimodal one (restarts and global search help) and a smooth valley
    # (sustained local refinement helps). No single runner wins both.
    problems = [
        Problem.from_benchmark("Rastrigin", Rastrigin(dimensions=DIMENSIONS)),
        Problem.from_benchmark("Rosenbrock", Rosenbrock(dimensions=DIMENSIONS)),
    ]
    algorithms = build_algorithms()

    # One grid. The four runners are structurally different; Benchmark treats
    # them identically because they share the name/color/run() contract. Same
    # seed => same x0 for every runner (Problem.starting_point), so the
    # comparison is fair by construction.
    bench = Benchmark(
        problems=problems,
        algorithms=algorithms,
        seeds=list(range(NUM_SEEDS)),
        output_dir=OUTPUT_DIR,
        num_workers=4,
        save_artifacts=True,  # dumps traces.json / runs.csv / summary.csv
    )
    print(
        f"Running {len(problems)}x{len(algorithms)}x{NUM_SEEDS} = "
        f"{len(problems) * len(algorithms) * NUM_SEEDS} jobs..."
    )
    bench.run(verbose=False)
    bench.print_summary()

    # Headline figure: multi-seed convergence — median best-fitness with a
    # 25/75 IQR band per runner, one panel per problem. The handoff marker
    # and the custom runner's restart steps are read straight off
    # bench.traces; the one call is identical for all four runners.
    plot_benchmark_convergence(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=f"Multi-seed convergence — median + IQR over {NUM_SEEDS} seeds (10-D)",
        save_path=OUTPUT_DIR / "01_heterogeneous_convergence.png",
    )

    # Persistence round-trip: the run dumped traces.json above; reloading it
    # reconstructs every RunTrace, so a re-plot or post-hoc analysis never
    # re-runs an optimizer.
    reloaded = load_traces_json(OUTPUT_DIR / "traces.json")
    n_in_memory = sum(len(v) for v in bench.traces.values())
    n_on_disk = sum(len(v) for v in reloaded.values())
    assert n_in_memory == n_on_disk == len(problems) * len(algorithms) * NUM_SEEDS
    print(
        f"\nPersistence round-trip OK: {n_on_disk} traces reloaded from "
        f"{OUTPUT_DIR / 'traces.json'} (no optimizer re-run)."
    )

    print(f"\nOutput written to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
