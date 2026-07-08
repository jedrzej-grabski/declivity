"""End-to-end demo of the declarative plotting API.

Runs three algorithms on a 10D Sphere and exercises:

1. ``plot_metrics(result)`` — every panel registered for that algorithm.
2. ``plot_metrics(result, panels=[...])`` — explicit panel selection.
3. ``plot_comparison({label: result})`` — overlay across algorithms with
   the panel set inferred from the registry intersection.
4. Introspection of the registry — what's available, what's common.

Output goes to ``plots/basic/declarative_plotting/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from declivity import AlgorithmFactory
from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.des.config import DESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.plotting import PanelRegistry, plot_comparison, plot_metrics
from declivity.utils.benchmark_functions import Sphere


plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/basic/declarative_plotting")
COLORS = {
    "CMA-ES":   "#e74c3c",
    "DES":      "#f39c12",
    "L-BFGS-B": "#3498db",
}


def _run(algorithm: AlgorithmChoice, x0: np.ndarray, sphere: Sphere):
    """Build a config with deep diagnostics on and run the algorithm."""
    if algorithm == AlgorithmChoice.CMAES:
        config: CMAESConfig | DESConfig | LBFGSBConfig = CMAESConfig(
            dimensions=len(x0)
        )
    elif algorithm == AlgorithmChoice.DES:
        config = DESConfig(dimensions=len(x0))
    elif algorithm == AlgorithmChoice.LBFGSB:
        config = LBFGSBConfig(dimensions=len(x0))
    else:
        raise ValueError(algorithm)

    # Turn on every diagnostic so the panels have something to draw.
    config.enable_all_diagnostics()

    optimizer = AlgorithmFactory.create_optimizer(
        algorithm=algorithm,
        func=sphere,
        initial_point=x0,
        config=config,
        lower_bounds=-100.0,
        upper_bounds=100.0,
        seed=42,
    )
    return optimizer.optimize()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sphere = Sphere(dimensions=10)
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-50.0, 50.0, size=10)

    print("Running CMA-ES, DES, L-BFGS-B on 10D Sphere...")
    cmaes_result = _run(AlgorithmChoice.CMAES, x0, sphere)
    des_result = _run(AlgorithmChoice.DES, x0, sphere)
    lbfgsb_result = _run(AlgorithmChoice.LBFGSB, x0, sphere)

    print(f"  CMA-ES   final: {cmaes_result.best_fitness:.3e}  ({cmaes_result.evaluations} evals)")
    print(f"  DES      final: {des_result.best_fitness:.3e}  ({des_result.evaluations} evals)")
    print(f"  L-BFGS-B final: {lbfgsb_result.best_fitness:.3e}  ({lbfgsb_result.evaluations} evals)")

    # 1. Single-algorithm deep dive — every panel registered for CMA-ES.
    print("\n1. plot_metrics: CMA-ES full diagnostics")
    plot_metrics(
        cmaes_result,
        title="CMA-ES on Sphere — every registered panel",
        save_path=OUTPUT_DIR / "01_cmaes_all_panels.png",
    )

    # 2. Single-algorithm with explicit panel selection (5 panels by key).
    selected = ["convergence", "step_size", "condition_number", "mean_norm", "det_covariance"]
    print(f"\n2. plot_metrics: CMA-ES selected {selected}")
    plot_metrics(
        cmaes_result,
        panels=selected,
        title="CMA-ES on Sphere — 5 hand-picked panels",
        save_path=OUTPUT_DIR / "02_cmaes_selected_panels.png",
    )

    # 3a. Side-by-side: CMA-ES vs DES (lots of shared panels).
    cmaes_des = PanelRegistry.common([AlgorithmChoice.CMAES, AlgorithmChoice.DES])
    print(f"\n3a. plot_comparison(CMA-ES vs DES) — common keys: {cmaes_des}")
    plot_comparison(
        {"CMA-ES": cmaes_result, "DES": des_result},
        colors=COLORS,
        title="CMA-ES vs DES on Sphere (common panels)",
        save_path=OUTPUT_DIR / "03a_cmaes_vs_des_common.png",
    )

    # 3b. Side-by-side: CMA-ES vs L-BFGS-B (very few shared panels, by design).
    cmaes_lbfgsb = PanelRegistry.common(
        [AlgorithmChoice.CMAES, AlgorithmChoice.LBFGSB]
    )
    print(f"\n3b. plot_comparison(CMA-ES vs L-BFGS-B) — common keys: {cmaes_lbfgsb}")
    plot_comparison(
        {"CMA-ES": cmaes_result, "L-BFGS-B": lbfgsb_result},
        colors=COLORS,
        title="CMA-ES vs L-BFGS-B on Sphere (common panels)",
        save_path=OUTPUT_DIR / "03b_cmaes_vs_lbfgsb_common.png",
    )

    # 4. Three-way comparison — what's available across all population-style algorithms?
    three_way_keys = PanelRegistry.common(
        [AlgorithmChoice.CMAES, AlgorithmChoice.DES, AlgorithmChoice.LBFGSB]
    )
    print(f"\n4. plot_comparison(all 3) — common keys: {three_way_keys}")
    if three_way_keys:
        plot_comparison(
            {"CMA-ES": cmaes_result, "DES": des_result, "L-BFGS-B": lbfgsb_result},
            colors=COLORS,
            ncols=1,
            title="Three-way comparison on Sphere",
            save_path=OUTPUT_DIR / "04_three_way_common.png",
        )

    # 5. Introspection — print every registered panel by algorithm.
    print("\n5. Registry contents")
    for algorithm, keys in PanelRegistry.all_registered().items():
        print(f"  {algorithm.value:8s}: {keys}")

    print(f"\nOutput written to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
