"""Constrained Rosenbrock demo — ConstraintHandler API end-to-end.

Demonstrates two constraint configurations on 2D Rosenbrock with DES, CMA-ES,
and L-BFGS-B:

(i)  BoxConstraintHandler(CLAMP) — box-only baseline, box bounds [-2, 3]².
(ii) PenaltyConstraintHandler  — box repair + quadratic penalty on
     g(x) = x₀² + x₁² - 1.5 ≤ 0 (disk constraint).

The unconstrained Rosenbrock optimum is (1, 1) which lies *outside* the disk
(‖(1,1)‖² = 2 > 1.5), so the penalty configuration steers the optimizer to the
feasible boundary, producing visibly different convergence behaviour.

Output goes to ``plots/basic/constrained_rosenbrock/``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import override

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from declivity import AlgorithmFactory
from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.des.config import DESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.plotting import plot_comparison
from declivity.utils import (
    BoxConstraintHandler,
    BoxStrategy,
    ConstraintHandler,
    Rosenbrock,
)

plt.ioff()
plt.switch_backend("Agg")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("plots/basic/constrained_rosenbrock")

# ---------------------------------------------------------------------------
# Box bounds  (tighter than the Rosenbrock default [-5, 10] for visual clarity)
# ---------------------------------------------------------------------------

DIM = 2
LOWER = np.full(DIM, -2.0)
UPPER = np.full(DIM, 3.0)

# ---------------------------------------------------------------------------
# PenaltyConstraintHandler — box repair + quadratic penalty for g(x) ≤ 0
# ---------------------------------------------------------------------------


class PenaltyConstraintHandler(ConstraintHandler):
    """Composes a BoxConstraintHandler for repair with a quadratic penalty term.

    The inequality constraint g(x) ≤ 0 is enforced *softly* by augmenting
    the objective::

        penalty(x, f_x) = f_x + coeff * max(0, g(x))²

    Repair is fully delegated to the wrapped box handler; the penalty does
    not perform any projection onto the inequality constraint boundary.
    """

    def __init__(
        self,
        box_handler: BoxConstraintHandler,
        g: Callable[[NDArray[np.float64]], float],
        penalty_coeff: float = 1e4,
    ) -> None:
        self._box = box_handler
        self._g = g
        self._coeff = penalty_coeff

    # ------------------------------------------------------------------
    # ConstraintHandler interface
    # ------------------------------------------------------------------

    @override
    def is_feasible(self, x: NDArray[np.float64]) -> bool:
        return self._box.is_feasible(x) and self._g(x) <= 0.0

    @override
    def feasibility_distance(self, x: NDArray[np.float64]) -> float:
        box_dist = self._box.feasibility_distance(x)
        ineq_violation = max(0.0, float(self._g(x)))
        return box_dist + ineq_violation**2

    @override
    def repair(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Delegate box repair; inequality violation is handled by penalty."""
        return self._box.repair(x)

    @override
    def penalty(self, x: NDArray[np.float64], f_x: float) -> float:
        """Add quadratic penalty for inequality violations."""
        violation = max(0.0, float(self._g(x)))
        return f_x + self._coeff * violation**2


# ---------------------------------------------------------------------------
# Inequality constraint: disk constraint g(x) = x₀² + x₁² - 1.5 ≤ 0
#
# Feasible region: closed disk of radius sqrt(1.5) ≈ 1.22 centred at origin.
# Unconstrained Rosenbrock optimum (1, 1): ‖(1,1)‖² = 2 > 1.5  →  infeasible.
# The constrained optimum therefore lies on the boundary arc near (1, 1).
# ---------------------------------------------------------------------------


def disk_constraint(x: NDArray[np.float64]) -> float:
    """g(x) = x₀² + x₁² - 1.5 ≤ 0."""
    return float(x[0] ** 2 + x[1] ** 2 - 1.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLORS = {
    "CMA-ES": "#e74c3c",
    "DES": "#f39c12",
    "L-BFGS-B": "#3498db",
}


def _make_config(
    algorithm: AlgorithmChoice,
) -> CMAESConfig | DESConfig | LBFGSBConfig:
    if algorithm is AlgorithmChoice.CMAES:
        cfg: CMAESConfig | DESConfig | LBFGSBConfig = CMAESConfig(dimensions=DIM)
    elif algorithm is AlgorithmChoice.DES:
        cfg = DESConfig(dimensions=DIM)
    elif algorithm is AlgorithmChoice.LBFGSB:
        cfg = LBFGSBConfig(dimensions=DIM)
    else:
        raise ValueError(algorithm)
    cfg.enable_all_diagnostics()
    return cfg


def _run_algorithm(
    algorithm: AlgorithmChoice,
    x0: NDArray[np.float64],
    rosenbrock: Rosenbrock,
    handler: ConstraintHandler,
    seed: int,
):
    config = _make_config(algorithm)
    return AlgorithmFactory.create_optimizer(
        algorithm=algorithm,
        func=rosenbrock,
        initial_point=x0,
        config=config,
        constraint_handler=handler,
        lower_bounds=LOWER,
        upper_bounds=UPPER,
        seed=seed,
    ).optimize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rosenbrock = Rosenbrock(dimensions=DIM)
    rng = np.random.default_rng(7)
    x0: NDArray[np.float64] = rng.uniform(LOWER, UPPER)

    print(f"2D Rosenbrock, box bounds {LOWER} … {UPPER}")
    print(f"Starting point x0 = {x0}")
    print("Unconstrained global optimum: (1, 1)  →  f = 0")
    print("Disk constraint g(x) = ‖x‖² - 1.5 ≤ 0")
    print(
        f"  g(1,1) = {disk_constraint(np.array([1.0, 1.0])):.3f}  "
        f"(> 0 → infeasible under penalty config)\n"
    )

    # ------------------------------------------------------------------
    # Configuration (i): box-only — BoxConstraintHandler(CLAMP)
    # ------------------------------------------------------------------

    box_handler = BoxConstraintHandler(BoxStrategy.CLAMP, LOWER, UPPER)

    print("=== Config (i): BoxConstraintHandler(CLAMP) — box-only baseline ===")
    box_results: dict[str, object] = {}
    for algo, label in [
        (AlgorithmChoice.CMAES, "CMA-ES"),
        (AlgorithmChoice.DES, "DES"),
        (AlgorithmChoice.LBFGSB, "L-BFGS-B"),
    ]:
        result = _run_algorithm(algo, x0, rosenbrock, box_handler, seed=42)
        box_results[label] = result
        print(
            f"  {label:8s}  best_f={result.best_fitness:.4e}"
            f"  x*={np.round(result.best_solution, 4)}"
            f"  evals={result.evaluations}"
        )

    plot_comparison(
        box_results,  # type: ignore[arg-type]
        colors=COLORS,
        title="Constrained Rosenbrock — Box-only (CLAMP)",
        save_path=OUTPUT_DIR / "01_box_only_comparison.png",
    )
    print("  → saved 01_box_only_comparison.png\n")

    # ------------------------------------------------------------------
    # Configuration (ii): PenaltyConstraintHandler — box + disk constraint
    # ------------------------------------------------------------------

    penalty_handler = PenaltyConstraintHandler(
        box_handler, disk_constraint, penalty_coeff=1e4
    )

    print("=== Config (ii): PenaltyConstraintHandler — box + disk constraint ===")
    penalty_results: dict[str, object] = {}
    for algo, label in [
        (AlgorithmChoice.CMAES, "CMA-ES"),
        (AlgorithmChoice.DES, "DES"),
        (AlgorithmChoice.LBFGSB, "L-BFGS-B"),
    ]:
        result = _run_algorithm(algo, x0, rosenbrock, penalty_handler, seed=42)
        penalty_results[label] = result
        g_val = disk_constraint(result.best_solution)
        feasible_str = "feasible" if g_val <= 0 else "infeasible"
        print(
            f"  {label:8s}  best_f={result.best_fitness:.4e}"
            f"  x*={np.round(result.best_solution, 4)}"
            f"  evals={result.evaluations}"
            f"  g(x*)={g_val:.4f} ({feasible_str})"
        )

    plot_comparison(
        penalty_results,  # type: ignore[arg-type]
        colors=COLORS,
        title="Constrained Rosenbrock — Penalty (box + disk g(x)≤0)",
        save_path=OUTPUT_DIR / "02_penalty_comparison.png",
    )
    print("  → saved 02_penalty_comparison.png\n")

    # ------------------------------------------------------------------
    # Sanity check: penalty config produces a different optimum than box-only
    # ------------------------------------------------------------------

    print("=== Convergence difference check ===")
    for label in ["CMA-ES", "DES", "L-BFGS-B"]:
        box_f = box_results[label].best_fitness  # type: ignore[union-attr]
        pen_f = penalty_results[label].best_fitness  # type: ignore[union-attr]
        delta = abs(pen_f - box_f)
        verdict = "DIFFERENT ✓" if delta > 1e-6 else "SAME (warn)"
        print(
            f"  {label:8s}  box_f={box_f:.4e}  penalty_f={pen_f:.4e}  Δ={delta:.4e}  {verdict}"
        )

    print(f"\nOutput written to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
