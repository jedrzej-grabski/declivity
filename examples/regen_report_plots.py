"""Regenerate the 5 report plots from saved traces.

Reads `traces.json` from each `plots/report/<panel>/` directory and re-renders
the convergence + boxplot at a single, consistent visual style suitable for
the supervisor report. No re-running of optimizers needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from src.benchmarking.persistence import load_traces_json
from src.benchmarking.plotter import BenchmarkPlotter
from src.benchmarking.problem import Problem
from src.benchmarking.algorithm_run import AlgorithmRun, SingleAlgorithm
from src.algorithms.choices import AlgorithmChoice

# Algorithm-name -> display color (keep stable across all report plots)
COLOR_MAP = {
    "CMA-ES":                          "#e74c3c",
    "L-BFGS-B":                        "#3498db",
    "CMA-ES -> L-BFGS-B (C^-1)":       "#2ecc71",
    "CMA-ES -> L-BFGS-B (identity)":   "#9b59b6",
    "Handoff (C^-1)":                  "#2ecc71",
    "Handoff (identity)":              "#9b59b6",
}


@dataclass
class StubAlgorithm:
    """Minimal AlgorithmRun-shaped object for re-plotting (no `run` method).

    Plotter only needs ``name`` and ``color`` from each algorithm spec, so
    we don't need to reconstruct the actual config_factory closures.
    """
    name: str
    color: str

    def run(self, *a, **kw):  # pragma: no cover - never called during replot
        raise NotImplementedError


def stub_problem(name: str, dimensions: int) -> Problem:
    """Plotter only needs ``name`` and ``dimensions`` from Problem; everything
    else is bypassed because we already have the traces."""
    return Problem(
        name=name,
        function=lambda x: 0.0,
        dimensions=dimensions,
        lower_bound=-1.0,
        upper_bound=1.0,
    )


def regen_panel(
    traces_dir: Path,
    save_dir: Path,
    title: str,
    boxplot_title: str | None = None,
    panel_aspect: tuple[float, float] = (10.0, 6.5),
    legend_fontsize: int = 11,
    floor: float = 1e-12,
) -> None:
    """Re-render convergence and final-fitness boxplot for a single panel."""
    save_dir.mkdir(parents=True, exist_ok=True)
    traces_file = traces_dir / "traces.json"

    traces = load_traces_json(traces_file)
    # Order matters for the legend. Use the order we encounter the algorithm
    # names. Sort so reference algorithms come first.
    seen_names: list[str] = []
    problem_dims: dict[str, int] = {}
    for (problem_name, algorithm_name), runs in traces.items():
        if algorithm_name not in seen_names:
            seen_names.append(algorithm_name)
        if runs:
            # Infer dimension from anywhere -- the problem name has it embedded
            # in our convention (e.g. "Rippled-c1000000-a0.1-rot-d50-m5").
            # Default to 0 if we can't parse it.
            try:
                token = next(t for t in problem_name.split("-") if t.startswith("d"))
                problem_dims[problem_name] = int(token[1:])
            except (StopIteration, ValueError):
                problem_dims[problem_name] = 0

    # Reorder: CMA-ES, L-BFGS-B, then handoffs (C^-1 first, identity second).
    priority = {
        "CMA-ES": 0,
        "L-BFGS-B": 1,
    }
    def name_key(name: str) -> tuple[int, str]:
        if name in priority:
            return (priority[name], name)
        if "C^-1" in name:
            return (10, name)
        if "identity" in name:
            return (11, name)
        return (20, name)
    seen_names.sort(key=name_key)

    algorithms = [
        StubAlgorithm(name=n, color=COLOR_MAP.get(n, "#7f8c8d"))
        for n in seen_names
    ]

    problems = [
        stub_problem(p_name, problem_dims.get(p_name, 0))
        for p_name in sorted(set(p for p, _ in traces.keys()))
    ]

    plotter = BenchmarkPlotter(
        problems=problems,
        algorithms=algorithms,
        traces=traces,
        output_dir=save_dir,
        floor=floor,
    )

    plotter.plot_convergence_grid(
        save_path=save_dir / "convergence.png",
        title=title,
        figsize_per_panel=panel_aspect,
        legend_fontsize=legend_fontsize,
    )

    if boxplot_title is not None:
        plotter.plot_final_fitness_boxplot(
            save_path=save_dir / "final_fitness.png",
            title=boxplot_title,
        )

    plt.close("all")
    print(f"  Wrote {save_dir}/convergence.png" + (
        f" + {save_dir}/final_fitness.png" if boxplot_title else ""
    ))


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    base = Path("plots/report")

    # ---- Plot 0: baseline (unrotated multimodal) ----
    # Re-loads from the per-problem subdirs; this benchmark wrote two panels.
    regen_combined_baseline(base)

    # ---- Plot 1: rotated near-unimodal, three dimensions ----
    regen_combined_low_amp(base)

    # ---- Plot 2: pure rotated ellipsoid (reproduction of old finding) ----
    regen_panel(
        traces_dir=base / "02_reproduce_old" / "Rippled-c1000000-a0-rot-d50-m5",
        save_dir=base / "02_reproduce_old",
        title="Pure rotated Ellipsoid d=50, cond=10⁶, m=5  —  loose L-BFGS-B termination",
        boxplot_title="Final fitness (pure rotated Ellipsoid)",
        panel_aspect=(11.0, 6.5),
    )

    # ---- Plot 3: multimodal anisotropic with tight budget ----
    regen_panel(
        traces_dir=base / "03_multimodal_tight" / "Rippled-c1000000-a1-rot-d50-m5",
        save_dir=base / "03_multimodal_tight",
        title="Rotated RippledEllipsoid d=50, cond=10⁶, amp=1, m=5  —  warmup 5000 / L-BFGS-B 1000",
        boxplot_title="Final fitness (multimodal, tight budget)",
        panel_aspect=(11.0, 6.5),
    )

    # ---- Plot 4: warmup timing sweep ----
    regen_panel(
        traces_dir=base / "04_timing_sweep",
        save_dir=base / "04_timing_sweep",
        title="Warmup timing sweep  —  rotated RippledEllipsoid d=30, cond=10⁶, amp=0.1, m=5",
        boxplot_title=None,
        panel_aspect=(12.0, 7.0),
        legend_fontsize=9,
    )

    print("All report plots regenerated.")


def regen_combined_baseline(base: Path) -> None:
    """Combine Rastrigin and Griewank baseline traces into one 2-panel figure."""
    combined: dict = {}
    problem_dims: dict[str, int] = {}
    seen_names: list[str] = []

    for problem_name in ("rastrigin", "griewank"):
        path = base / "00_baseline_unrotated" / problem_name / "traces.json"
        if not path.exists():
            print(f"  [skip] missing {path}")
            return
        per_problem = load_traces_json(path)
        for (p, a), runs in per_problem.items():
            combined.setdefault((p, a), []).extend(runs)
            if a not in seen_names:
                seen_names.append(a)
            problem_dims[p] = runs[0].evaluations and 10 or 10  # known d=10

    def name_key(name: str) -> tuple[int, str]:
        if name == "CMA-ES":           return (0, name)
        if name == "L-BFGS-B":         return (1, name)
        if "C^-1" in name:             return (10, name)
        if "identity" in name:         return (11, name)
        return (20, name)
    seen_names.sort(key=name_key)

    algorithms = [StubAlgorithm(name=n, color=COLOR_MAP.get(n, "#7f8c8d")) for n in seen_names]

    problem_names = sorted(set(p for p, _ in combined.keys()))
    problems = [stub_problem(p, problem_dims.get(p, 10)) for p in problem_names]

    plotter = BenchmarkPlotter(
        problems=problems,
        algorithms=algorithms,
        traces=combined,
        output_dir=base / "00_baseline_unrotated",
    )
    plotter.plot_convergence_grid(
        save_path=base / "00_baseline_unrotated" / "convergence.png",
        title="Baseline: unrotated Rastrigin and Griewank, d=10  —  identity matches C⁻¹",
        figsize_per_panel=(9.0, 6.5),
        legend_fontsize=10,
    )
    plotter.plot_final_fitness_boxplot(
        save_path=base / "00_baseline_unrotated" / "final_fitness.png",
        title="Final fitness (baseline, unrotated)",
    )
    plt.close("all")
    print(f"  Wrote {base/'00_baseline_unrotated'}/convergence.png + boxplot")


def regen_combined_low_amp(base: Path) -> None:
    """Combine d=10, d=30, d=50 traces of rotated low-ripple RippledEllipsoid."""
    combined: dict = {}
    problem_dims: dict[str, int] = {}
    seen_names: list[str] = []

    panels = [
        ("Rippled-c1000000-a0.1-rot-d10-m5", 10),
        ("Rippled-c1000000-a0.1-rot-d30-m5", 30),
        ("Rippled-c1000000-a0.1-rot-d50-m5", 50),
    ]
    for subdir_name, d in panels:
        path = base / "01_rippled_low_amp" / subdir_name / "traces.json"
        if not path.exists():
            print(f"  [skip] missing {path}")
            return
        per = load_traces_json(path)
        for (p, a), runs in per.items():
            combined.setdefault((p, a), []).extend(runs)
            if a not in seen_names:
                seen_names.append(a)
            problem_dims[p] = d

    def name_key(name: str) -> tuple[int, str]:
        if name == "CMA-ES":           return (0, name)
        if name == "L-BFGS-B":         return (1, name)
        if "C^-1" in name:             return (10, name)
        if "identity" in name:         return (11, name)
        return (20, name)
    seen_names.sort(key=name_key)

    algorithms = [StubAlgorithm(name=n, color=COLOR_MAP.get(n, "#7f8c8d")) for n in seen_names]

    problem_names = sorted(set(p for p, _ in combined.keys()),
                          key=lambda n: problem_dims.get(n, 0))
    problems = [stub_problem(p, problem_dims.get(p, 0)) for p in problem_names]

    plotter = BenchmarkPlotter(
        problems=problems,
        algorithms=algorithms,
        traces=combined,
        output_dir=base / "01_rippled_low_amp",
        floor=1e-22,  # values reach 1e-20 here; the default 1e-12 would clip C^-1
    )
    plotter.plot_convergence_grid(
        save_path=base / "01_rippled_low_amp" / "convergence.png",
        title="Rotated RippledEllipsoid, cond=10⁶, amp=0.1, m=5  —  C⁻¹ wins by orders of magnitude",
        figsize_per_panel=(8.0, 6.0),
        legend_fontsize=9,
    )
    plt.close("all")
    print(f"  Wrote {base/'01_rippled_low_amp'}/convergence.png")


if __name__ == "__main__":
    main()
