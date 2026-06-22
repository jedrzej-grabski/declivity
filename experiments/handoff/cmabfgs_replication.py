"""CMABFGS comparison: our CMA-ES ⇆ L-BFGS-B variant vs Maksym's CMA-ES ⇆ BFGS.

Reproduces the CMABFGS figure from Maksym's thesis (``notes/test_bfgs.png``;
local — ``notes/`` is gitignored) with our interleaved scheme as a variant that
swaps BFGS for L-BFGS-B. The reference figure compares, on
``d=100``, ``f = SDP`` with the optimum in a corner of the ``[-180, 20]^d``
box. The reference names its function "SDP" but defines it as the axis-aligned
ill-conditioned Ellipsoid ``f(x) = sum_i 10^(6 i/(d-1)) x_i^2`` (a convex,
separable quadratic, condition number ``10^6``), *not* the BBOB
ramping-exponent Different Powers — so we use :class:`Ellipsoid`. Contenders:

- standalone BFGS,
- standalone CMA-ES,
- **CMABFGS** — interleaved CMA-ES + BFGS — for handoff intervals
  ``N = k·d`` with ``k ∈ {0.5, 1, 2, 4, 8}``.

This script runs the same experiment with *our* algorithms — CMA-ES,
L-BFGS-B, and ``InterleavedCMAESLBFGSB`` (CMA-ES ⇆ **L-BFGS-B**) — so the
two figures can be compared side by side: does our CMA-ES⇆L-BFGS-B variant
behave like Maksym's CMA-ES⇆BFGS? Our figure lands in
``plots/handoff/cmabfgs_replication/`` (gitignored; regenerate via the run
commands below).

It is a thin, framework-conforming orchestration (see
``docs/NEW_CODE_building_an_experiment.md``):

- the problem is a ``Problem`` built from a ``BenchmarkFunction``;
- every contender is an ``AlgorithmRun`` (``SingleAlgorithm`` /
  ``InterleavedCMAESLBFGSB``) carrying its own ``name`` + ``color``;
- a single-seed ``Benchmark`` runs them and auto-persists ``traces.json``;
- the figure is the reusable ``plot_convergence_overlay`` (single-panel
  overlay with the reference's secondary "CMA-ES iterations" axis), which
  reads colours straight off the algorithm specs.

Standalone BFGS is opt-in (``--show-bfgs``): on this smooth bowl L-BFGS-B
alone converges in ~12k evals and visually dwarfs the CMA-ES-vs-CMABFGS
comparison the figure is about.

Population size is a CLI knob because the reference's secondary axis
(``1e6`` evals ↔ ~2500 iterations) implies ``λ ≈ 4·d = 400``; we run both
that and the framework default to see which matches.

Run::

    PYTHONPATH=. pdm run python experiments/handoff/cmabfgs_replication.py --popsize 400 --tag pop4d
    PYTHONPATH=. pdm run python experiments/handoff/cmabfgs_replication.py --popsize 0   --tag popdefault
    # re-render a figure from saved traces (no re-running):
    PYTHONPATH=. pdm run python experiments/handoff/cmabfgs_replication.py \
        --popsize 400 --tag pop4d --replot-from plots/handoff/cmabfgs_replication/pop4d/traces.json
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig, default_population_size
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.benchmarking import (
    AlgorithmRun,
    Benchmark,
    InterleavedCMAESLBFGSB,
    Problem,
    SingleAlgorithm,
    load_traces_json,
)
from src.plotting import plot_convergence_overlay
from src.utils.benchmark_functions import Ellipsoid, ShiftedFunction
from src.utils.initial_point_generator import UniformBoxInitialPointGenerator


plt.ioff()
plt.switch_backend("Agg")


# Palette chosen to echo the reference figure.
COLOR_BFGS = "#1f3d7a"
COLOR_CMAES = "#d62728"
K_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]
K_COLORS = {
    0.5: "#aec7e8",   # light blue
    1.0: "#ff7f0e",   # orange
    2.0: "#ffbb78",   # light orange
    4.0: "#2ca02c",   # dark green
    8.0: "#98df8a",   # light green
}


def build_problem(
    dimensions: int, lower: float, upper: float, corner_fraction: float
) -> Problem:
    """SDP = the reference's ill-conditioned Ellipsoid, optimum near the corner.

    The reference calls its test function "SDP" but defines it as the
    axis-aligned quadratic ``f(x) = sum_i 10^(6 i/(d-1)) x_i^2`` (condition
    number ``10^6``, constant diagonal Hessian) — the classic Ellipsoid, *not*
    the BBOB ramping-exponent Different Powers. We use :class:`Ellipsoid`
    accordingly.

    ``corner_fraction = 1.0`` lands the optimum exactly on the corner — but
    note that a *bounded* L-BFGS-B then solves it in a single step (the first
    Cauchy point projects every coordinate onto the active bound, which is
    the optimum), unlike the reference's *unbounded* BFGS. The default 0.8
    places the optimum at the *origin* (zero shift), i.e. 10% of the box
    width from the corner — exactly the reference's "corner with margin"
    case (case 2, box ``[-180, 20]``). Values keep the optimum interior, so
    L-BFGS-B has to descend the ill-conditioned bowl rather than snap to a
    bound.
    """
    base = Ellipsoid(dimensions, lower=lower, upper=upper)
    func = ShiftedFunction.near_corner(
        base, fraction=corner_fraction, name_suffix="SDP-corner"
    )
    # The reference draws the initial CMA-ES mean from U[-100, 100]^d
    # regardless of the (asymmetric) feasible box, so the start region is
    # centred on the optimum rather than on the box. Match that exactly.
    start = UniformBoxInitialPointGenerator(-100.0, 100.0)
    return Problem.from_benchmark("SDP", func, initial_point_generator=start)


def build_algorithms(
    dimensions: int,
    population_size: int,
    total_budget: int,
    memory_size: int,
    probe_max_evals: int,
    probe_factr: float,
    probe_pgtol: float,
    include_bfgs: bool,
) -> list[AlgorithmRun]:
    """The contenders, each carrying its own ``name`` + ``color``.

    Draw order = (optional BFGS) -> CMABFGS k=0.5..8 -> CMA-ES, so the CMA-ES
    reference curve sits on top in the overlay.
    """

    def cmaes_config(d: int) -> CMAESConfig:
        return CMAESConfig(
            dimensions=d, budget=total_budget, population_size=population_size
        )

    def lbfgsb_config(d: int) -> LBFGSBConfig:
        return LBFGSBConfig(
            dimensions=d, budget=total_budget, m=memory_size, pgtol=1e-10, factr=0
        )

    algorithms: list[AlgorithmRun] = []

    if include_bfgs:
        algorithms.append(
            SingleAlgorithm(
                name="BFGS", color=COLOR_BFGS,
                algorithm=AlgorithmChoice.LBFGSB, config_factory=lbfgsb_config,
            )
        )

    for k in K_VALUES:
        interval = max(1, round(k * dimensions))
        algorithms.append(
            InterleavedCMAESLBFGSB(
                name=f"CMABFGS, k={k:g}", color=K_COLORS[k],
                cmaes_config_factory=cmaes_config,
                lbfgsb_config_factory=lbfgsb_config,
                cmaes_interval=interval,
                total_budget=total_budget,
                transform="inverse",
                probe_factr=probe_factr,
                probe_pgtol=probe_pgtol,
                probe_max_evals=probe_max_evals,
            )
        )

    algorithms.append(
        SingleAlgorithm(
            name="CMA-ES", color=COLOR_CMAES,
            algorithm=AlgorithmChoice.CMAES, config_factory=cmaes_config,
        )
    )
    return algorithms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=100)
    parser.add_argument("--lower", type=float, default=-180.0)
    parser.add_argument("--upper", type=float, default=20.0)
    parser.add_argument("--total-budget", type=int, default=1_000_000)
    parser.add_argument(
        "--popsize", type=int, default=400,
        help="CMA-ES population size lambda. 0 = framework default (4 + floor(3 ln d)). "
             "400 = 4*d, matching the reference's iteration axis.",
    )
    parser.add_argument("--memory-size", type=int, default=10)
    parser.add_argument(
        "--corner-fraction", type=float, default=0.8,
        help="Where the optimum sits between box centre (0) and corner (1). "
             "0.8 = the origin (zero shift), 10%% of the box width from the corner, "
             "matching the reference's case 2; 1.0 = exactly on the corner, which a "
             "bounded L-BFGS-B solves in one step.",
    )
    parser.add_argument(
        "--probe-max-evals", type=int, default=20000,
        help="Hard cap on evaluations per L-BFGS-B burst (generous, so a burst "
             "can descend the ill-conditioned quadratic to full convergence).",
    )
    parser.add_argument(
        "--probe-factr", type=float, default=10.0,
        help="Burst relative-decrease stop. Small (~10) => run each burst to "
             "full convergence (like the reference's BFGS-to-convergence); "
             "large (1e7) => 'stops advancing rapidly'.",
    )
    parser.add_argument("--probe-pgtol", type=float, default=1e-12)
    parser.add_argument(
        "--show-bfgs", action="store_true",
        help="Include the standalone BFGS (L-BFGS-B) baseline. Off by default: "
             "on this smooth bowl L-BFGS-B alone is so dominant it overshadows "
             "the CMA-ES-vs-CMABFGS comparison the figure is about.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Parallel workers; one contender per worker. Results are identical "
             "to serial (each run is independent and deterministic).",
    )
    parser.add_argument("--floor", type=float, default=1e-13)
    parser.add_argument("--tag", type=str, default=None,
                        help="Filename suffix, e.g. pop4d / popdefault.")
    parser.add_argument(
        "--replot-from", type=Path, default=None,
        help="Skip the runs and re-render the figure from a saved Benchmark "
             "traces.json (algorithm specs are rebuilt for colours/names).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("plots/handoff/cmabfgs_replication"),
    )
    args = parser.parse_args()

    effective_lambda = (
        args.popsize if args.popsize > 0 else default_population_size(args.dimensions)
    )
    tag = args.tag or (f"pop{args.popsize}" if args.popsize > 0 else "popdefault")

    print(
        f"\nSDP replication: d={args.dimensions}, box=[{args.lower:g}, {args.upper:g}], "
        f"lambda={effective_lambda}, budget={args.total_budget}, "
        f"k={K_VALUES} (interval=k*d)"
    )

    # Problem + algorithm specs are cheap to build and are needed in both the
    # run path (to execute) and the replot path (for colours / names / title).
    problem = build_problem(args.dimensions, args.lower, args.upper, args.corner_fraction)
    algorithms = build_algorithms(
        args.dimensions, args.popsize, args.total_budget, args.memory_size,
        args.probe_max_evals, args.probe_factr, args.probe_pgtol, args.show_bfgs,
    )

    if args.replot_from is not None:
        traces = load_traces_json(args.replot_from)
        print(f"  re-plotting from {args.replot_from}")
    else:
        # Single-seed Benchmark: still gives same-seed x0 and auto-persists
        # traces.json / runs.csv / summary.csv into a per-tag subdirectory.
        bench = Benchmark(
            problems=[problem], algorithms=algorithms,
            seeds=[args.seed], output_dir=args.output_dir / tag,
            num_workers=args.num_workers,
        )
        bench.run(verbose=True)
        bench.print_summary()
        traces = bench.traces

    save_path = args.output_dir / f"cmabfgs_{tag}.png"
    plot_convergence_overlay(
        traces, problem, algorithms,
        title=(
            f"d={args.dimensions}, f=SDP\n"
            f"optimum w rogu obszaru dopuszczalnego (granice: "
            f"[{args.lower:g}, {args.upper:g}]$^d$)"
        ),
        xlabel="Liczba ewaluacji funkcji celu  (function evaluations)",
        ylabel="Best fitness (log)",
        floor=args.floor,
        secondary_iter_lambda=effective_lambda,
        secondary_label=f"Iteracje CMA-ES  (CMA-ES iterations, λ={effective_lambda})",
        legend_loc="upper right",
        save_path=save_path,
    )
    print(f"\nSaved: {save_path.absolute()}")


if __name__ == "__main__":
    main()
