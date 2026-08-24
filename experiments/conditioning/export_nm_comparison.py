"""Export the lean-vs-Hessian Nelder-Mead comparison from persisted exp1 traces.

exp1 draws one figure *per arm* (all conditioners overlaid), which answers "which
conditioner is best for this arm" but not "does passing the Hessian beat plain
Nelder-Mead at all".  This reads the same ``traces.parquet`` back and re-cuts it the
other way: one panel per conditioner, the four Nelder-Mead arms overlaid, with
the lean baseline drawn identically in every panel as the reference line.

The four arms are a 2x2 over how the conditioner reaches the run (see
``LOCAL_VARIANTS`` in ``common.py``)::

                        no model step          model step
    isotropic simplex   neldermead_control     neldermead_hc
    shaped simplex      neldermead             neldermead_hc_shaped

``neldermead_control`` is handed the conditioner and ignores it, so its curve is
the *same run* in every panel -- both the honest baseline and a check that the
arms are otherwise identical.

Writes to ``plots/conditioning/exp1/<study>/<...>/nm_comparison/``:

    arms_by_conditioner.png   one panel per conditioner, four arms overlaid
    arms_best.png             single-panel view of the strongest conditioners
    summary.csv               the 2x2 as median / min / max per (arm, conditioner)

With more than one ``--dim`` it also writes an aggregate under
``.../nm_comparison_aggregate/``.  Raw gaps cannot be averaged across
dimensions -- they differ by orders of magnitude -- so the aggregate is built
from the **paired** per-seed ratio to that seed's own control run:

    gain(arm, conditioner, dim, seed) = log10( control_gap / arm_gap )

which reads directly as "orders of magnitude below lean Nelder-Mead", is
dimensionless, and pairs every arm against the exact baseline run it should be
compared with (same dimension, same seed, same starting point, same rotation)
rather than against a median of a different sample.

Usage::

    PYTHONPATH=. uv run python -m experiments.conditioning.export_nm_comparison \\
        --study-name nmhc_demo --dim 10 --variant unbounded --num-seeds 5

    PYTHONPATH=. uv run python -m experiments.conditioning.export_nm_comparison \\
        --study-name nmhc_multi --dim 10 20 30 --num-seeds 25
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from declivity.benchmarking import RunTrace, load_traces_parquet
from experiments.conditioning.common import apply_dark_style
from experiments.conditioning.exp1_conditioners import (
    Exp1Spec,
    benchmark_dir,
    conditioners_for,
    contender_name,
    family_for,
)

plt.ioff()
plt.switch_backend("Agg")

# Arm key -> (label as it appears in the trace store, colour, line style).
ARMS: list[tuple[str, str, str, str]] = [
    ("neldermead_control", "Nelder-Mead (control)", "#9aa0a6", "--"),
    ("neldermead", "Nelder-Mead", "#f5a623", "-"),
    ("neldermead_hc", "Nelder-Mead HC", "#39c0ed", "-"),
    ("neldermead_hc_shaped", "Nelder-Mead HC + simplex", "#37d67a", "-"),
]
ARM_DESCRIPTIONS = {
    "neldermead_control": "conditioner ignored (lean Nelder-Mead)",
    "neldermead": "conditioner shapes the initial simplex only",
    "neldermead_hc": "conditioner drives the model step only",
    "neldermead_hc_shaped": "conditioner does both",
}


def median_curve(
    traces: list[RunTrace], grid: np.ndarray, floor: float
) -> np.ndarray | None:
    """Median best-so-far across seeds on a shared evaluation grid.

    Each trace is held flat past its own final evaluation rather than dropped,
    so a run that converged early keeps contributing its (good) final value to
    the median instead of silently shrinking the sample.
    """
    if not traces:
        return None
    curves = []
    for trace in traces:
        evaluations = np.asarray(trace.evaluations, dtype=float)
        values = np.maximum(np.asarray(trace.best_fitness, dtype=float), floor)
        if evaluations.size == 0:
            continue
        curves.append(
            np.interp(grid, evaluations, values, left=values[0], right=values[-1])
        )
    return np.median(np.vstack(curves), axis=0) if curves else None


def collect(
    spec: Exp1Spec, scaling: str, variant: str, dim: int
) -> tuple[dict[tuple[str, str], list[RunTrace]], str, float]:
    path = benchmark_dir(spec, scaling, variant, dim) / "traces.parquet"
    if not path.exists():
        raise SystemExit(
            f"No traces at {path} -- run exp1 for this cell first, e.g.\n"
            f"  PYTHONPATH=. uv run python -m experiments.conditioning.exp1_hydra "
            f"experiment=demo objective={spec.objective} dim={dim} "
            f"variant={variant} study_name={spec.name}"
        )
    family = family_for(spec, variant, dim)
    template = family.template
    optimum = 0.0
    minimum = getattr(template.function, "global_minimum", None)
    if minimum is not None:
        optimum = float(minimum[1])
    return load_traces_parquet(path), template.name, optimum


def figure_by_conditioner(
    traces: dict[tuple[str, str], list[RunTrace]],
    problem_name: str,
    spec: Exp1Spec,
    dim: int,
    optimum: float,
    floor: float,
    save_path: Path,
) -> None:
    conditioners = conditioners_for(spec, dim)
    columns = 4
    rows = int(np.ceil(len(conditioners) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 3.9 * rows), squeeze=False
    )

    for index, conditioner in enumerate(conditioners):
        ax = axes[index // columns][index % columns]
        finals: dict[str, float] = {}
        for key, label, colour, style in ARMS:
            runs = traces.get((problem_name, contender_name(key, conditioner)), [])
            if not runs:
                continue
            horizon = max(t.evaluations[-1] for t in runs if t.evaluations)
            grid = np.linspace(1.0, horizon, 400)
            shifted = [
                RunTrace(
                    algorithm=t.algorithm,
                    problem=t.problem,
                    seed=t.seed,
                    evaluations=t.evaluations,
                    best_fitness=[max(f - optimum, floor) for f in t.best_fitness],
                    final_evaluations=t.final_evaluations,
                    final_fitness=max(t.final_fitness - optimum, floor),
                )
                for t in runs
            ]
            curve = median_curve(shifted, grid, floor)
            if curve is None:
                continue
            ax.plot(
                grid,
                curve,
                color=colour,
                linestyle=style,
                lw=2.4,
                label=label,
                zorder=3,
            )
            finals[label] = float(np.median([t.final_fitness for t in shifted]))

        baseline = finals.get("Nelder-Mead (control)")
        title = f"{conditioner.label}"
        if baseline is not None:
            best = min(finals.values())
            if best < baseline:
                title += f"   (best arm {baseline / max(best, floor):.0f}x below lean)"
        ax.set_title(title, fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel(r"$f(x_{best}) - f^*$")
        ax.grid(alpha=0.2, which="both")

    for index in range(len(conditioners), rows * columns):
        axes[index // columns][index % columns].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        [f"{label}  —  {ARM_DESCRIPTIONS[key]}" for (key, label, _, _) in ARMS],
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=False,
    )
    fig.suptitle(
        f"Lean Nelder-Mead vs. passing it the geometry — {problem_name}, "
        f"{spec.num_seeds} seeds (median). The grey dashed baseline is the same "
        f"run in every panel.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {save_path.name}")


def figure_best(
    traces: dict[tuple[str, str], list[RunTrace]],
    problem_name: str,
    spec: Exp1Spec,
    dim: int,
    optimum: float,
    floor: float,
    save_path: Path,
) -> None:
    """Single panel: the lean baseline against each arm's own best conditioner."""
    conditioners = conditioners_for(spec, dim)
    fig, ax = plt.subplots(figsize=(12, 6.8))

    horizon = 0
    for _, _, runs in _arm_runs(traces, problem_name, conditioners):
        horizon = max(
            horizon, max((t.evaluations[-1] for t in runs if t.evaluations), default=0)
        )
    grid = np.linspace(1.0, max(horizon, 2), 500)

    palette = {key: colour for key, _, colour, _ in ARMS}
    styles = {key: style for key, _, _, style in ARMS}
    for key, label, _, _ in ARMS:
        best_conditioner, best_value, best_runs = None, np.inf, []
        for conditioner in conditioners:
            runs = traces.get((problem_name, contender_name(key, conditioner)), [])
            if not runs:
                continue
            value = float(
                np.median([max(t.final_fitness - optimum, floor) for t in runs])
            )
            if value < best_value:
                best_conditioner, best_value, best_runs = conditioner, value, runs
        if best_conditioner is None:
            continue
        shifted = [
            RunTrace(
                algorithm=t.algorithm,
                problem=t.problem,
                seed=t.seed,
                evaluations=t.evaluations,
                best_fitness=[max(f - optimum, floor) for f in t.best_fitness],
                final_evaluations=t.final_evaluations,
                final_fitness=max(t.final_fitness - optimum, floor),
            )
            for t in best_runs
        ]
        curve = median_curve(shifted, grid, floor)
        if curve is None:
            continue
        suffix = (
            "" if key == "neldermead_control" else f" | best {best_conditioner.label}"
        )
        ax.plot(
            grid,
            curve,
            color=palette[key],
            linestyle=styles[key],
            lw=2.8,
            label=f"{label}{suffix}  →  {best_value:.2e}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("function evaluations")
    ax.set_ylabel(r"$f(x_{best}) - f^*$")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=10, loc="lower left")
    ax.set_title(
        f"Each arm at its own best conditioner — {problem_name}, "
        f"{spec.num_seeds} seeds (median)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {save_path.name}")


def _arm_runs(traces, problem_name, conditioners):
    for key, _, _, _ in ARMS:
        for conditioner in conditioners:
            runs = traces.get((problem_name, contender_name(key, conditioner)), [])
            if runs:
                yield key, conditioner, runs


def write_csv(
    traces: dict[tuple[str, str], list[RunTrace]],
    problem_name: str,
    spec: Exp1Spec,
    dim: int,
    optimum: float,
    floor: float,
    save_path: Path,
) -> None:
    conditioners = conditioners_for(spec, dim)
    baseline: dict[str, float] = {}
    rows = []
    for key, label, _, _ in ARMS:
        for conditioner in conditioners:
            runs = traces.get((problem_name, contender_name(key, conditioner)), [])
            if not runs:
                continue
            finals = np.array(
                [max(t.final_fitness - optimum, floor) for t in runs], dtype=float
            )
            evaluations = np.array([t.final_evaluations for t in runs], dtype=float)
            row = {
                "arm": key,
                "arm_label": label,
                "arm_meaning": ARM_DESCRIPTIONS[key],
                "conditioner": conditioner.key,
                "seeds": len(runs),
                "median_gap": float(np.median(finals)),
                "min_gap": float(finals.min()),
                "max_gap": float(finals.max()),
                "median_evaluations": float(np.median(evaluations)),
            }
            if key == "neldermead_control":
                baseline[conditioner.key] = row["median_gap"]
            rows.append(row)

    reference = float(np.median(list(baseline.values()))) if baseline else float("nan")
    for row in rows:
        row["speedup_vs_lean"] = (
            reference / row["median_gap"] if row["median_gap"] > 0 else float("inf")
        )

    with save_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {save_path.name}  (lean baseline gap = {reference:.4e})")

    width = max(len(label) for _, label, _, _ in ARMS) + 2
    print(
        f"\n{'conditioner':<14}"
        + "".join(f"{label:>{width}}" for _, label, _, _ in ARMS)
    )
    print("-" * (14 + width * len(ARMS)))
    for conditioner in conditioners:
        line = f"{conditioner.key:<14}"
        for key, _, _, _ in ARMS:
            match = [
                r
                for r in rows
                if r["arm"] == key and r["conditioner"] == conditioner.key
            ]
            line += (
                f"{match[0]['median_gap']:>{width}.3e}" if match else f"{'-':>{width}}"
            )
        print(line)


# Aggregation across dimensions


def paired_gains(
    traces: dict[tuple[str, str], list[RunTrace]],
    problem_name: str,
    spec: Exp1Spec,
    dim: int,
    optimum: float,
    floor: float,
) -> list[dict[str, object]]:
    """Per-seed ``log10(control_gap / arm_gap)`` for every (arm, conditioner).

    Pairing is what makes this aggregatable: seed ``s`` at dimension ``d`` is
    compared against *its own* control run, which shares the starting point and
    the rotation, so the ratio isolates the mechanism rather than the instance.
    """
    control_key = ARMS[0][0]
    control: dict[int, float] = {}
    for conditioner in conditioners_for(spec, dim):
        for trace in traces.get(
            (problem_name, contender_name(control_key, conditioner)), []
        ):
            # Identical across conditioners by construction; first wins.
            control.setdefault(trace.seed, max(trace.final_fitness - optimum, floor))

    rows: list[dict[str, object]] = []
    for key, label, _, _ in ARMS:
        for conditioner in conditioners_for(spec, dim):
            for trace in traces.get(
                (problem_name, contender_name(key, conditioner)), []
            ):
                reference = control.get(trace.seed)
                if reference is None:
                    continue
                gap = max(trace.final_fitness - optimum, floor)
                rows.append(
                    {
                        "dim": dim,
                        "seed": trace.seed,
                        "arm": key,
                        "arm_label": label,
                        "conditioner": conditioner.key,
                        "conditioner_label": conditioner.label,
                        "gap": gap,
                        "control_gap": reference,
                        "log10_gain": float(np.log10(reference / gap)),
                    }
                )
    return rows


def figure_aggregate(
    rows: list[dict[str, object]],
    spec: Exp1Spec,
    dims: list[int],
    save_path: Path,
) -> None:
    """Orders of magnitude below lean Nelder-Mead, pooled over seeds and dims."""
    conditioners = [c.key for c in conditioners_for(spec, dims[0])]
    labels = {c.key: c.label for c in conditioners_for(spec, dims[0])}
    positions = np.arange(len(conditioners), dtype=float)
    arms = ARMS[1:]  # the control is the reference line at 0
    width = 0.8 / len(arms)

    fig, axes = plt.subplots(
        1, 2, figsize=(19, 6.4), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    ax = axes[0]
    for index, (key, label, colour, _) in enumerate(arms):
        medians, lows, highs = [], [], []
        for conditioner in conditioners:
            gains = np.array(
                [
                    r["log10_gain"]
                    for r in rows
                    if r["arm"] == key and r["conditioner"] == conditioner
                ],
                dtype=float,
            )
            if gains.size == 0:
                medians.append(np.nan)
                lows.append(0.0)
                highs.append(0.0)
                continue
            median = float(np.median(gains))
            medians.append(median)
            lows.append(median - float(np.percentile(gains, 25)))
            highs.append(float(np.percentile(gains, 75)) - median)
        offset = (index - (len(arms) - 1) / 2) * width
        ax.bar(
            positions + offset,
            medians,
            width=width * 0.92,
            color=colour,
            label=label,
            yerr=[lows, highs],
            ecolor="#ffffff",
            capsize=3,
            error_kw={"alpha": 0.45, "lw": 1.0},
        )
    ax.axhline(0.0, color="#9aa0a6", lw=2.0, ls="--")
    ax.text(
        len(conditioners) - 0.45,
        0.15,
        "lean Nelder-Mead",
        color="#9aa0a6",
        fontsize=9,
        ha="right",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[c] for c in conditioners])
    ax.set_xlabel("conditioner")
    ax.set_ylabel("orders of magnitude below lean Nelder-Mead\n(median, IQR)")
    ax.set_title(
        f"Pooled over {spec.num_seeds} seeds x d in {{{', '.join(map(str, dims))}}}",
        fontsize=11,
    )
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=9, loc="upper left")

    # Right panel: does the effect hold as dimension grows?
    ax = axes[1]
    for key, label, colour, style in arms:
        best_by_dim = []
        for dim in dims:
            gains = [
                r["log10_gain"] for r in rows if r["arm"] == key and r["dim"] == dim
            ]
            # Best conditioner for this arm at this dimension, by median gain.
            per_conditioner = [
                float(
                    np.median(
                        [
                            r["log10_gain"]
                            for r in rows
                            if r["arm"] == key
                            and r["dim"] == dim
                            and r["conditioner"] == conditioner
                        ]
                        or [np.nan]
                    )
                )
                for conditioner in conditioners
            ]
            best_by_dim.append(np.nanmax(per_conditioner) if gains else np.nan)
        ax.plot(
            dims,
            best_by_dim,
            "o-",
            color=colour,
            linestyle=style,
            lw=2.6,
            ms=7,
            label=label,
        )
    ax.axhline(0.0, color="#9aa0a6", lw=2.0, ls="--")
    ax.set_xticks(dims)
    ax.set_xlabel("dimension")
    ax.set_ylabel("orders of magnitude below lean\n(best conditioner, median)")
    ax.set_title("Each arm at its best conditioner, per dimension", fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "How much does passing Nelder-Mead the geometry actually buy? "
        "Paired against each seed's own lean run.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {save_path.name}")


def write_aggregate_csv(rows: list[dict[str, object]], save_path: Path) -> None:
    with save_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {save_path.name}  ({len(rows)} paired runs)")


def print_aggregate(
    rows: list[dict[str, object]], spec: Exp1Spec, dims: list[int]
) -> None:
    conditioners = conditioners_for(spec, dims[0])
    arms = ARMS[1:]
    width = 26
    print(
        f"\nMedian orders of magnitude below lean Nelder-Mead "
        f"(pooled over seeds and d in {dims})\n"
    )
    print(
        f"{'conditioner':<14}" + "".join(f"{label:>{width}}" for _, label, _, _ in arms)
    )
    print("-" * (14 + width * len(arms)))
    for conditioner in conditioners:
        line = f"{conditioner.key:<14}"
        for key, _, _, _ in arms:
            gains = [
                r["log10_gain"]
                for r in rows
                if r["arm"] == key and r["conditioner"] == conditioner.key
            ]
            line += f"{np.median(gains):>+{width}.2f}" if gains else f"{'-':>{width}}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", default="nmhc_demo")
    parser.add_argument("--objective", default="ellipsoid")
    parser.add_argument(
        "--dim",
        type=int,
        nargs="+",
        default=[10],
        help="One or more dimensions. Several also writes the paired aggregate.",
    )
    parser.add_argument("--variant", default="unbounded")
    parser.add_argument("--scaling", default="none")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--snapshot-ks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--rotate", action="store_true", default=True)
    parser.add_argument("--floor", type=float, default=1e-22)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    apply_dark_style()
    dims = sorted(set(args.dim))
    aggregate_rows: list[dict[str, object]] = []

    def spec_for(dim: int) -> Exp1Spec:
        return Exp1Spec(
            name=args.study_name,
            objective=args.objective,
            dimensions=(dim,),
            num_seeds=args.num_seeds,
            variants=(args.variant,),
            rotate=args.rotate,
            snapshot_ks=tuple(args.snapshot_ks),
            scalings=(args.scaling,),
        )

    def out_dir(spec: Exp1Spec, dim: int, leaf: str) -> Path:
        return (
            spec.plot_root
            / spec.name
            / f"rot{int(spec.rotate)}"
            / args.scaling
            / (args.variant)
            / f"d{dim:03d}"
            / leaf
        )

    for dim in dims:
        spec = spec_for(dim)
        traces, problem_name, optimum = collect(spec, args.scaling, args.variant, dim)
        out = args.output_dir or out_dir(spec, dim, "nm_comparison")
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n{problem_name}, d={dim}, {args.variant}, scaling={args.scaling}")

        figure_by_conditioner(
            traces,
            problem_name,
            spec,
            dim,
            optimum,
            args.floor,
            out / "arms_by_conditioner.png",
        )
        figure_best(
            traces, problem_name, spec, dim, optimum, args.floor, out / "arms_best.png"
        )
        write_csv(
            traces, problem_name, spec, dim, optimum, args.floor, out / "summary.csv"
        )
        aggregate_rows.extend(
            paired_gains(traces, problem_name, spec, dim, optimum, args.floor)
        )
        print(f"  -> {out}")

    if len(dims) > 1 and aggregate_rows:
        spec = spec_for(dims[0])
        root = (
            spec.plot_root
            / spec.name
            / f"rot{int(spec.rotate)}"
            / args.scaling
            / args.variant
            / "nm_comparison_aggregate"
        )
        root.mkdir(parents=True, exist_ok=True)
        print(f"\nAggregate over d in {dims}")
        figure_aggregate(aggregate_rows, spec, dims, root / "aggregate_gain.png")
        write_aggregate_csv(aggregate_rows, root / "aggregate.csv")
        print_aggregate(aggregate_rows, spec, dims)
        print(f"\nexported -> {root}")


if __name__ == "__main__":
    main()
