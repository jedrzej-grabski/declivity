# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

"""Interactive browser for exp2 hybrid (CMA-ES + local optimizer) results.

Renders convergence overlays and suite ECDFs on the fly from persisted
``traces.parquet`` files (never from pre-saved PNGs). Every ``CMA+<opt>,
k=...`` curve is already fully spliced by ``exp2_hybrid.py``'s compose stage
(``compose_switch_trace``) at batch-run time and written straight into
``traces.parquet`` -- this notebook only loads and picks curves by name, it
never re-splices.

Three views:

- View 1 -- per-optimizer detail: one optimizer's k-sensitivity (alone vs.
  CMA+opt at every k vs. CMA-ES), for one (scaling, dim, function).
- View 2 -- cross-optimizer at fixed k: one curve per optimizer (its
  ``CMA+<opt>, k`` hybrid at a chosen switch interval) on one (scaling, dim,
  function), to compare optimizers head-to-head.
- View 3 -- cross-optimizer suite ECDF at fixed k: the same cross-optimizer
  comparison aggregated over every function in the study, via
  ``plot_suite_ecdf``.

Launch with:

    PYTHONPATH=. uv run marimo edit experiments/conditioning/visualize_exp2.py
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")

with app.setup:
    import sys
    from dataclasses import dataclass, replace
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, to_rgba
    from matplotlib.figure import Figure

    # Ensure the project root is on the path regardless of cwd.
    _ROOT = Path(__file__).parent.parent.parent.resolve()
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from declivity.benchmarking import RunTrace, load_traces_parquet
    from declivity.plotting import plot_convergence_overlay, plot_suite_ecdf
    from experiments.conditioning.common import (
        CMAES_COLOR,
        CurveSpec,
        LOCAL_LABELS,
        anchor_traces,
        apply_dark_style,
        gap_traces,
        load_yaml,
        plot_xmax,
        problem_optimum,
        ramp_colors,
        resolve_population_size,
    )
    from experiments.conditioning.exp2_hybrid import (
        Exp2Spec,
        benchmark_dir,
        contenders_for,
        family_for,
        hybrid_name,
    )

    DATA_ROOT = Path("results/conditioning/exp2")
    FLOOR = 1e-9
    # plot_convergence_overlay/plot_suite_ecdf default to a 16x9in figure,
    # sized for a standalone saved PNG. That is wider than a browser
    # viewport, so a hstack of several defaults to one-per-row regardless of
    # `wrap` -- shrinking here is what actually gets panels side by side.
    FIGSIZE_GRID = (7.0, 4.5)  # View 1: several panels per row.
    FIGSIZE_SINGLE = (11.0, 6.5)


@app.class_definition
@dataclass
class StudyInfo:
    """One discovered ``<data_root>/<name>`` study directory.

    ``cells`` is ``{scaling: {dim: (function_numbers...)}}``, read straight
    off the ``benchmarks/`` tree rather than trusted from ``study.yaml``:
    exp2's Hydra sweep axes are ``dim`` and ``function_number`` (one SLURM
    array task = one compute cell), so a shared ``study.yaml`` written by the
    last-finishing cell under-reports the realized grid (mirrors exp1's
    ``visualize.py`` doing the same for its own sweep axes).
    """

    name: str
    study_dir: Path
    spec: Exp2Spec
    cells: dict[str, dict[int, tuple[int, ...]]]


@app.function
def spec_from_study_yaml(study_dir: Path) -> Exp2Spec:
    """Reconstruct an ``Exp2Spec`` from a persisted ``study.yaml``.

    ``dimensions`` is left empty here (never recorded in ``study.yaml``, see
    ``Exp2Spec.study_descriptor``); ``discover_studies`` fills it from
    ``cells``.
    """
    payload = load_yaml(study_dir / "study.yaml")
    return Exp2Spec(
        name=payload["name"],
        edition=payload["edition"],
        functions=tuple(payload.get("functions", ())),
        dimensions=(),
        num_seeds=payload["num_seeds"],
        variant=payload["variant"],
        rotate=payload["rotate"],
        ks=tuple(payload["ks"]),
        granularity=payload["granularity"],
        cmaes_evaluations_per_dim=payload["cmaes_evaluations_per_dim"],
        population_factor=payload["population_factor"],
        sigma0=payload["sigma0"],
        probe_budget_per_dim=payload["probe_budget_per_dim"],
        optimizers=tuple(payload["optimizers"]),
        transform=payload["transform"],
        scalings=tuple(payload["scalings"]),
        data_root=study_dir.parent,
    )


@app.function
def discover_cells(
    study_dir: Path, scalings: tuple[str, ...], variant: str
) -> dict[str, dict[int, tuple[int, ...]]]:
    """``{scaling: {dim: (function_numbers with a traces.parquet...)}}``."""
    cells: dict[str, dict[int, tuple[int, ...]]] = {}
    for scaling in scalings:
        bench_root = study_dir / scaling / "benchmarks" / variant
        if not bench_root.is_dir():
            continue
        by_dim: dict[int, tuple[int, ...]] = {}
        for dim_dir in sorted(bench_root.iterdir()):
            if not (dim_dir.is_dir() and dim_dir.name.startswith("d")):
                continue
            functions = tuple(
                sorted(
                    int(fn_dir.name[1:])
                    for fn_dir in dim_dir.iterdir()
                    if fn_dir.is_dir()
                    and fn_dir.name.startswith("f")
                    and (fn_dir / "traces.parquet").exists()
                )
            )
            if functions:
                by_dim[int(dim_dir.name[1:])] = functions
        if by_dim:
            cells[scaling] = by_dim
    return cells


@app.function
def discover_studies(data_root: Path) -> list[StudyInfo]:
    """Glob ``<data_root>/*/study.yaml`` and build one ``StudyInfo`` per match
    that has at least one realized ``(scaling, dim, function)`` cell."""
    infos = []
    for study_yaml in sorted(data_root.glob("*/study.yaml")):
        study_dir = study_yaml.parent
        spec = spec_from_study_yaml(study_dir)
        candidate_scalings = tuple(s for s in spec.scalings if (study_dir / s).is_dir())
        cells = discover_cells(study_dir, candidate_scalings, spec.variant)
        if not cells:
            continue
        dims = sorted({dim for by_dim in cells.values() for dim in by_dim})
        spec = replace(spec, dimensions=tuple(dims))
        infos.append(StudyInfo(name=spec.name, study_dir=study_dir, spec=spec, cells=cells))
    return infos


@app.function
def load_function_shifted(
    spec: Exp2Spec, scaling: str, dim: int, function_number: int, floor: float = FLOOR
):
    """Load + anchor/gap one function's ``traces.parquet``, mirroring
    ``exp2_hybrid.run_plot_stage``'s per-function pipeline. Returns
    ``(shifted_traces, template)`` or ``None`` if the file is missing."""
    traces_path = benchmark_dir(spec, scaling, dim, function_number) / "traces.parquet"
    if not traces_path.exists():
        return None
    traces = load_traces_parquet(traces_path)
    family = family_for(spec, dim, function_number)
    template = family.template
    optimum = problem_optimum(template)
    f0 = {
        seed: max(
            float(
                family.instance(seed).function(family.instance(seed).starting_point(seed))
            )
            - optimum,
            floor,
        )
        for seed in range(spec.num_seeds)
    }
    shifted = anchor_traces(gap_traces(traces, optimum, floor), f0)
    return shifted, template


@app.function
def load_suite_traces(
    spec: Exp2Spec, scaling: str, dim: int, functions: tuple[int, ...]
) -> tuple[dict[tuple[str, str], list[RunTrace]], list]:
    """Raw (un-anchored) traces + templates pooled over every function, for
    ``plot_suite_ecdf`` (which computes its own per-problem gap-to-optimum),
    mirroring ``run_plot_stage``'s suite-aggregation loop."""
    suite_traces: dict[tuple[str, str], list[RunTrace]] = {}
    templates = []
    for function_number in functions:
        traces_path = benchmark_dir(spec, scaling, dim, function_number) / "traces.parquet"
        if not traces_path.exists():
            continue
        suite_traces.update(load_traces_parquet(traces_path))
        templates.append(family_for(spec, dim, function_number).template)
    return suite_traces, templates


@app.function
def faded(color: str, alpha: float = 0.4) -> str:
    """A translucent variant of ``color`` (same hue) for baseline curves that
    should read as secondary to their full-opacity hybrid counterpart."""
    return to_hex(to_rgba(color, alpha=alpha), keep_alpha=True)


@app.function
def cross_optimizer_contenders(
    spec: Exp2Spec, k: float, show_alone: bool
) -> list[CurveSpec]:
    """One curve per optimizer's ``CMA+<opt>, k`` hybrid at a fixed switch
    interval, plus CMA-ES-alone as reference. Each optimizer keeps one color
    across both its hybrid (full opacity) and its stand-alone baseline
    (faded), when ``show_alone`` requests the baselines too."""
    colors = ramp_colors(len(spec.optimizers))
    specs = []
    if show_alone:
        specs += [
            CurveSpec(LOCAL_LABELS[optimizer_key], faded(color))
            for optimizer_key, color in zip(spec.optimizers, colors, strict=True)
        ]
    specs += [
        CurveSpec(hybrid_name(optimizer_key, k), color)
        for optimizer_key, color in zip(spec.optimizers, colors, strict=True)
    ]
    specs.append(CurveSpec("CMA-ES", CMAES_COLOR))
    return specs


@app.function
def render_panel(
    shifted,
    template,
    runners: list[CurveSpec],
    *,
    title: str,
    population_size: int,
    figsize: tuple[float, float] = FIGSIZE_SINGLE,
    floor: float = FLOOR,
) -> Figure | None:
    """One ``plot_convergence_overlay`` panel for an arbitrary contender
    list, with the secondary CMA-ES-iteration axis exp2 always shows."""
    pooled = [
        trace for runner in runners for trace in shifted.get((template.name, runner.name), [])
    ]
    if not pooled:
        return None
    return plot_convergence_overlay(
        shifted,
        template,
        runners,
        title=title,
        ylabel="$f(x_{best}) - f^*$",
        floor=floor,
        show_iqr=False,
        annotate_final=False,
        xmax=plot_xmax(pooled),
        secondary_iter_lambda=population_size,
        secondary_label="CMA-ES iterations",
        figsize=figsize,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    apply_dark_style()
    return


@app.cell
def _(mo):
    mo.md("""
    # exp2 hybrid browser
    "
        "Renders convergence overlays and suite ECDFs live from persisted "
        "`traces.parquet` files under `results/conditioning/exp2/`. Every "
        "`CMA+<opt>, k=...` curve was already spliced at batch-run time -- "
        "nothing here re-composes a hybrid.
    """)
    return


@app.cell
def _():
    studies = discover_studies(DATA_ROOT)
    return (studies,)


@app.cell(hide_code=True)
def _(mo, studies):
    mo.stop(
        not studies,
        mo.callout(
            mo.md(f"No studies found under `{DATA_ROOT}`. Run exp2 first."),
            kind="warn",
        ),
    )
    mo.md(f"Found {len(studies)} stud{'y' if len(studies) == 1 else 'ies'}.")
    return


@app.cell(hide_code=True)
def _(mo, studies):
    mo.stop(not studies)

    study_options = {s.name: s for s in studies}
    study_sel = mo.ui.dropdown(
        list(study_options.keys()), value=next(iter(study_options)), label="Study"
    )
    study_sel
    return study_options, study_sel


@app.cell(hide_code=True)
def _(mo, study_options, study_sel):
    mo.stop(study_sel.value is None)

    study = study_options[study_sel.value]
    scalings = list(study.cells.keys())
    scaling_sel = mo.ui.dropdown(scalings, value=scalings[0], label="Scaling")
    scaling_sel
    return scaling_sel, study


@app.cell(hide_code=True)
def _(mo, scaling_sel, study):
    mo.stop(scaling_sel.value is None)

    dims = sorted(study.cells[scaling_sel.value].keys())
    dim_sel = mo.ui.dropdown([str(d) for d in dims], value=str(dims[0]), label="Dimension")
    dim_sel
    return (dim_sel,)


@app.cell(hide_code=True)
def _(dim_sel, mo, scaling_sel, study):
    mo.stop(dim_sel.value is None)

    functions = study.cells[scaling_sel.value][int(dim_sel.value)]
    function_sel = mo.ui.dropdown(
        [str(f) for f in functions], value=str(functions[0]), label="Function"
    )
    function_sel
    return (function_sel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## View 1 -- per-optimizer detail
    "
        "One optimizer's k-sensitivity: alone vs. `CMA+opt, k` at every "
        "switch interval vs. CMA-ES alone, for one function.
    """)
    return


@app.cell(hide_code=True)
def _(dim_sel, function_sel, mo, scaling_sel, study):
    mo.stop(function_sel.value is None)

    v1_dim = int(dim_sel.value)
    v1_function = int(function_sel.value)
    v1_result = load_function_shifted(study.spec, scaling_sel.value, v1_dim, v1_function)
    mo.stop(
        v1_result is None,
        mo.callout(mo.md("No `traces.parquet` found for this selection."), kind="warn"),
    )
    assert v1_result is not None
    v1_shifted, v1_template = v1_result

    v1_population = resolve_population_size(v1_dim, study.spec.population_factor)
    v1_figs = []
    for v1_opt_key in study.spec.optimizers:
        v1_runners = contenders_for(study.spec, v1_opt_key)
        v1_fig = render_panel(
            v1_shifted,
            v1_template,
            v1_runners,
            title=f"{v1_template.name}, d={v1_dim}, CMA-ES + {LOCAL_LABELS[v1_opt_key]}",
            population_size=v1_population,
            figsize=FIGSIZE_GRID,
        )
        if v1_fig is not None:
            v1_figs.append(mo.as_html(v1_fig))
            plt.close(v1_fig)

    mo.stop(not v1_figs, mo.callout(mo.md("No optimizer curves to show."), kind="warn"))
    v1_per_row = 3
    v1_rows = [
        mo.hstack(v1_figs[i : i + v1_per_row], gap=1, widths="equal")
        for i in range(0, len(v1_figs), v1_per_row)
    ]
    mo.vstack(v1_rows, gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## View 2 -- cross-optimizer at fixed k
    "
        "Freeze a switch interval `k` and compare every optimizer's `CMA+opt, "
        "k` hybrid head-to-head, on one function.
    """)
    return


@app.cell(hide_code=True)
def _(mo, study):
    v2_ks = sorted(study.spec.ks)
    v2_k_sel = mo.ui.dropdown(
        [f"{k:g}" for k in v2_ks], value=f"{v2_ks[0]:g}", label="k"
    )
    v2_alone_sel = mo.ui.checkbox(value=False, label="Show stand-alone baselines")
    mo.hstack([v2_k_sel, v2_alone_sel], gap=2)
    return v2_alone_sel, v2_k_sel


@app.cell(hide_code=True)
def _(dim_sel, function_sel, mo, scaling_sel, study, v2_alone_sel, v2_k_sel):
    mo.stop(v2_k_sel.value is None or function_sel.value is None)

    v2_dim = int(dim_sel.value)
    v2_function = int(function_sel.value)
    v2_k = float(v2_k_sel.value)
    v2_result = load_function_shifted(study.spec, scaling_sel.value, v2_dim, v2_function)
    mo.stop(
        v2_result is None,
        mo.callout(mo.md("No `traces.parquet` found for this selection."), kind="warn"),
    )
    assert v2_result is not None
    v2_shifted, v2_template = v2_result

    v2_population = resolve_population_size(v2_dim, study.spec.population_factor)
    v2_runners = cross_optimizer_contenders(study.spec, v2_k, v2_alone_sel.value)
    v2_fig = render_panel(
        v2_shifted,
        v2_template,
        v2_runners,
        title=f"{v2_template.name}, d={v2_dim}, k={v2_k:g} -- cross-optimizer",
        population_size=v2_population,
    )
    mo.stop(v2_fig is None, mo.callout(mo.md("No curves to show."), kind="warn"))
    assert v2_fig is not None
    v2_out = mo.as_html(v2_fig)
    plt.close(v2_fig)
    v2_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## View 3 -- cross-optimizer suite ECDF at fixed k
    "
        "The same cross-optimizer comparison as View 2, aggregated over "
        "every function in the study for the chosen dimension -- which "
        "optimizer wins overall, at this switch interval.
    """)
    return


@app.cell(hide_code=True)
def _(mo, study):
    v3_ks = sorted(study.spec.ks)
    v3_k_sel = mo.ui.dropdown(
        [f"{k:g}" for k in v3_ks], value=f"{v3_ks[0]:g}", label="k"
    )
    v3_alone_sel = mo.ui.checkbox(value=False, label="Show stand-alone baselines")
    mo.hstack([v3_k_sel, v3_alone_sel], gap=2)
    return v3_alone_sel, v3_k_sel


@app.cell(hide_code=True)
def _(dim_sel, mo, scaling_sel, study, v3_alone_sel, v3_k_sel):
    mo.stop(v3_k_sel.value is None)

    v3_dim = int(dim_sel.value)
    v3_k = float(v3_k_sel.value)
    v3_functions = study.cells[scaling_sel.value][v3_dim]
    v3_suite_traces, v3_templates = load_suite_traces(
        study.spec, scaling_sel.value, v3_dim, v3_functions
    )
    mo.stop(not v3_templates, mo.callout(mo.md("No traces to aggregate."), kind="warn"))

    v3_runners = cross_optimizer_contenders(study.spec, v3_k, v3_alone_sel.value)
    v3_fig = plot_suite_ecdf(
        v3_suite_traces,
        v3_templates,
        v3_runners,
        title=(
            f"Suite ECDF, {study.spec.edition.upper()}, d={v3_dim}, "
            f"k={v3_k:g} -- cross-optimizer ({len(v3_templates)} functions)"
        ),
        show_subtitle=False,
        figsize=FIGSIZE_SINGLE,
    )
    v3_out = mo.as_html(v3_fig)
    plt.close(v3_fig)
    v3_out
    return


if __name__ == "__main__":
    app.run()
