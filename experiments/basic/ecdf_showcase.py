"""End-to-end demo of the target-hitting ECDF plot.

Runs a small (problem x algorithm x seed) benchmark and exercises:

1. ``plot_benchmark_ecdf`` — one curve per algorithm, showing what fraction
   of a shared set of log-spaced target levels each algorithm has reached by
   a given evaluation budget.
2. The derived ``global_minimum`` — left unset, so the value is read off
   ``problem.function.global_minimum``. Rastrigin and Sphere both have
   ``f* = 0``, so a second panel repeats Rastrigin on a shifted objective
   whose optimum is 500 to show the difference the gap makes: pass the wrong
   ``f*`` and every target below it is unreachable.
3. ``threshold_ceiling`` — a fixed target range, which is what makes curves
   comparable across separate figures rather than only within one.

Why this exists: an ECDF is easy to compute plausibly and wrongly. DES logs
its first iteration before its incumbent is set, so every DES trace starts
with ``+inf``; a single one of those in an unguarded target grid pins every
algorithm's curve to ~1.0 and the figure stops discriminating while still
looking reasonable. Keeping DES in this demo is the regression check.

Output goes to ``plots/basic/ecdf_showcase/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.des.config import DESConfig
from declivity.algorithms.powell.config import PowellConfig
from declivity.benchmarking import Benchmark, Problem, SingleAlgorithm
from declivity.benchmarking.ecdf import ecdf_auc
from declivity.plotting import plot_benchmark_ecdf
from declivity.utils.benchmark_functions import BenchmarkFunction, Rastrigin
from declivity.utils.stopping_conditions import MaxEvaluations

plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/basic/ecdf_showcase")

COLORS = {
    "DES": "#e74c3c",
    "CMA-ES": "#3498db",
    "Powell": "#2ecc71",
}

DIMENSIONS = 10
BUDGET = 6000
SEEDS = 10
BIAS = 500.0


class BiasedRastrigin(BenchmarkFunction):
    """Rastrigin lifted by a constant, so ``f*`` is 500 rather than 0.

    Stands in for the CEC suite, whose functions carry a ``100·i`` bias. An
    ECDF built against raw fitness on a problem like this cannot reach any
    target below the bias.
    """

    def __init__(self, dimensions: int, bias: float = BIAS):
        super().__init__(dimensions)
        self._inner = Rastrigin(dimensions)
        self._bias = bias

    def __call__(self, x: NDArray[np.float64]) -> float:
        return self._inner(x) + self._bias

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._inner.bounds

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        optimum, _ = self._inner.global_minimum
        return optimum, self._bias


def build_algorithms() -> list[SingleAlgorithm]:
    return [
        SingleAlgorithm(
            name="DES",
            color=COLORS["DES"],
            algorithm=AlgorithmChoice.DES,
            config_factory=lambda d: DESConfig(dimensions=d),
            stopping_condition=MaxEvaluations(BUDGET),
        ),
        SingleAlgorithm(
            name="CMA-ES",
            color=COLORS["CMA-ES"],
            algorithm=AlgorithmChoice.CMAES,
            config_factory=lambda d: CMAESConfig(dimensions=d),
            stopping_condition=MaxEvaluations(BUDGET),
        ),
        SingleAlgorithm(
            name="Powell",
            color=COLORS["Powell"],
            algorithm=AlgorithmChoice.POWELL,
            config_factory=lambda d: PowellConfig(dimensions=d),
            stopping_condition=MaxEvaluations(BUDGET),
        ),
    ]


def report(label: str, figure: plt.Figure) -> None:
    axes = figure.axes[0]
    print(f"\n{label}")
    for line in axes.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        print(
            f"  {str(line.get_label()):26s} "
            f"start={y[0]:.3f}  end={y[-1]:.3f}  span={y.max() - y.min():.3f}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plain = Problem.from_benchmark("Rastrigin", Rastrigin(dimensions=DIMENSIONS))
    biased = Problem.from_benchmark(
        "Rastrigin+500", BiasedRastrigin(dimensions=DIMENSIONS)
    )
    algorithms = build_algorithms()

    bench = Benchmark(
        problems=[plain, biased],
        algorithms=algorithms,
        seeds=list(range(SEEDS)),
        output_dir=OUTPUT_DIR / "_bench",
        num_workers=1,
        save_artifacts=False,
    )
    bench.run()

    figure = plot_benchmark_ecdf(
        bench.traces,
        plain,
        algorithms,
        title="Targets reached vs. budget",
        save_path=OUTPUT_DIR / "ecdf_rastrigin.png",
    )
    report("Rastrigin, f* read from the problem (0.0):", figure)

    # Same runs, same problem, correct f*=500 vs. the 0.0 someone would pass
    # by hand. The wrong f* makes every target below the bias unreachable.
    correct = plot_benchmark_ecdf(
        bench.traces,
        biased,
        algorithms,
        title="Shifted objective, f* = 500 (correct)",
        save_path=OUTPUT_DIR / "ecdf_biased_correct.png",
    )
    report("Rastrigin+500, f* derived correctly (500.0):", correct)

    wrong = plot_benchmark_ecdf(
        bench.traces,
        biased,
        algorithms,
        global_minimum=0.0,
        title="Shifted objective, f* forced to 0 (wrong)",
        save_path=OUTPUT_DIR / "ecdf_biased_wrong.png",
    )
    report("Rastrigin+500, f* forced to 0.0 (wrong):", wrong)

    # A fixed target range is what makes two figures comparable; the default
    # data-derived ceiling moves whenever the pooled runs change.
    fixed = plot_benchmark_ecdf(
        bench.traces,
        plain,
        algorithms,
        threshold_ceiling=1e2,
        title="Fixed target range [1e-8, 1e2]",
        save_path=OUTPUT_DIR / "ecdf_fixed_range.png",
    )
    report("Rastrigin, fixed target range:", fixed)

    # AUC is integrated in the log budget domain, matching the drawn axis.
    axes = figure.axes[0]
    print("\nAUC per algorithm (log-budget domain):")
    for line in axes.get_lines():
        x = np.asarray(line.get_xdata(), dtype=np.int64)
        y = np.asarray(line.get_ydata(), dtype=float)
        print(f"  {str(line.get_label()):26s} {ecdf_auc(x, y):.4f}")

    print(f"\nOutput written to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
