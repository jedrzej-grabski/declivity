# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

"""Interactive browser for exp1 conditioning-study results.

Renders convergence overlays on the fly from persisted ``traces.parquet`` files
(never from pre-saved PNGs), so every plot stays interactive. Mirrors the
load -> anchor -> render pipeline in
``experiments/conditioning/exp1_conditioners.py::run_plot_stage``.

Launch with:

    PYTHONPATH=. uv run marimo edit experiments/conditioning/visualize.py
"""

from __future__ import annotations

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")

with app.setup:
    import sys
    from dataclasses import dataclass, replace
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    # Ensure the project root is on the path regardless of cwd.
    _ROOT = Path(__file__).parent.parent.parent.resolve()
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from declivity.benchmarking import ConditionedLocalAlgorithm, load_traces_parquet
    from declivity.plotting import plot_convergence_overlay
    from experiments.conditioning.common import (
        CEC_OBJECTIVE,
        ELLIPSOID_OBJECTIVE,
        LOCAL_LABELS,
        anchor_traces,
        apply_dark_style,
        filter_seed,
        gap_traces,
        load_yaml,
        plot_xmax,
        problem_optimum,
    )
    from experiments.conditioning.exp1_conditioners import (
        Exp1Spec,
        benchmark_dir,
        build_contenders,
        family_for,
    )

    DATA_ROOT = Path("results/conditioning/exp1")
    FLOOR = 1e-9


@app.class_definition
@dataclass
class StudyInfo:
    """One discovered ``<name>[/f{NN}]/rot{0,1}`` study directory."""

    name: str
    objective: str
    function_number: int
    rotate: int
    study_dir: Path
    spec: Exp1Spec
    scalings: tuple[str, ...]  # only the scalings that actually exist on disk


@app.function
def spec_from_study_yaml(study_dir: Path) -> Exp1Spec:
    """Reconstruct an ``Exp1Spec`` from a persisted ``study.yaml``.

    ``study_dir`` is ``<data_root>/<name>[/f{NN}]/rot{0,1}`` (the function
    segment only exists for CEC studies, see ``study_root`` in
    ``exp1_conditioners.py``); ``data_root`` is recovered by walking up that
    many levels. Lists become tuples to match the spec's field types.
    """
    payload = load_yaml(study_dir / "study.yaml")
    depth = 3 if payload["objective"] == CEC_OBJECTIVE else 2
    data_root = study_dir
    for _ in range(depth):
        data_root = data_root.parent
    return Exp1Spec(
        name=payload["name"],
        objective=payload["objective"],
        edition=payload["edition"],
        function_number=payload["function_number"],
        # study.yaml omits the sweep axes (see Exp1Spec.study_descriptor);
        # discover_studies fills these from the on-disk benchmarks/ tree.
        dimensions=tuple(payload.get("dimensions", ())),
        num_seeds=payload["num_seeds"],
        variants=tuple(payload.get("variants", ())),
        rotate=payload["rotate"],
        snapshot_ks=tuple(payload["snapshot_ks"]),
        cmaes_evaluations_per_dim=payload["cmaes_evaluations_per_dim"],
        include_hessian=payload["include_hessian"],
        include_identity=payload["include_identity"],
        optimizers=tuple(payload["optimizers"]),
        transform=payload["transform"],
        scalings=tuple(payload["scalings"]),
        population_factor=payload["population_factor"],
        sigma0=payload["sigma0"],
        local_budget_per_dim=payload["local_budget_per_dim"],
        data_root=data_root,
    )


def _variants_and_dims_on_disk(
    study_dir: Path, scalings: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Actual ``(variant, dim)`` cells with a ``benchmarks/`` tree, read
    straight off disk.

    ``study.yaml`` is shared across every ``(dim, variant)`` Hydra sweep cell
    and deliberately omits these sweep axes (see
    ``Exp1Spec.study_descriptor``), so scanning ``benchmarks/`` is the only
    source of truth for which cells actually exist.
    """
    variants: set[str] = set()
    dims: set[int] = set()
    for scaling in scalings:
        bench_root = study_dir / scaling / "benchmarks"
        if not bench_root.is_dir():
            continue
        for variant_dir in bench_root.iterdir():
            if not variant_dir.is_dir():
                continue
            variants.add(variant_dir.name)
            for dim_dir in variant_dir.iterdir():
                if dim_dir.is_dir() and dim_dir.name.startswith("d"):
                    dims.add(int(dim_dir.name[1:]))
    return tuple(sorted(variants)), tuple(sorted(dims))


@app.function
def discover_studies(data_root: Path) -> list[StudyInfo]:
    """Glob ``<data_root>/*/rot*/study.yaml`` and ``<data_root>/*/f*/rot*/study.yaml``
    and build one ``StudyInfo`` per match.

    The two depths correspond to ellipsoid studies (no function segment) and
    CEC studies (``f{NN}`` segment) respectively -- see ``study_root`` in
    ``exp1_conditioners.py``. ``scalings``/``variants``/``dimensions`` are
    taken from what is actually present on disk rather than from
    ``study.yaml``: a study.yaml can list a scaling whose local/plot stage
    hasn't run yet, and it omits the variants/dimensions sweep axes entirely
    (see ``_variants_and_dims_on_disk``). Studies with no realized
    ``(dim, variant)`` cells are skipped.
    """
    infos = []
    study_yamls = {
        *data_root.glob("*/rot*/study.yaml"),
        *data_root.glob("*/f*/rot*/study.yaml"),
    }
    for study_yaml in sorted(study_yamls):
        study_dir = study_yaml.parent
        spec = spec_from_study_yaml(study_dir)
        scalings = tuple(s for s in spec.scalings if (study_dir / s).is_dir())
        variants, dims = _variants_and_dims_on_disk(study_dir, scalings)
        if not variants or not dims:
            # A study.yaml exists but no local/benchmarks cells are realized
            # yet (e.g. only CMA-ES has run); nothing to plot, so skip it.
            continue
        spec = replace(spec, variants=variants, dimensions=dims)
        infos.append(
            StudyInfo(
                name=spec.name,
                objective=spec.objective,
                function_number=spec.function_number,
                rotate=int(spec.rotate),
                study_dir=study_dir,
                spec=spec,
                scalings=scalings,
            )
        )
    return infos


@app.function
def group_by_name(studies: list[StudyInfo]) -> dict[str, dict[int, StudyInfo]]:
    """``key -> {rotate: StudyInfo}``, for View 2's rotate/scaling/variant
    sweeps that need to look across both rotations of the same study.

    ``key`` is ``name`` for ellipsoid studies (a single fixed function) and
    ``name f{NN}`` for CEC studies, so different functions sharing the same
    study name (e.g. a Hydra sweep over ``function_number`` with no
    ``--study-name``) don't collide into one entry.
    """
    by_name: dict[str, dict[int, StudyInfo]] = {}
    for info in studies:
        key = (
            info.name
            if info.objective == ELLIPSOID_OBJECTIVE
            else f"{info.name} f{info.function_number:02d}"
        )
        by_name.setdefault(key, {})[info.rotate] = info
    return by_name


@app.function
def load_shifted(
    spec: Exp1Spec, variant: str, dim: int, scaling: str, floor: float = FLOOR
):
    """Load ``traces.parquet`` for one (variant, dim, scaling) cell and replicate
    ``run_plot_stage``'s anchor/gap pipeline: gap-to-optimum, then anchor
    every curve at its seed's shared ``f(x0)`` so contenders visibly start
    from one point. Returns ``None`` if the traces file doesn't exist (e.g.
    the local/plot stage hasn't been run for this facet combination).

    Returns ``(shifted_traces, template_problem, contenders)`` where
    ``contenders`` is ``{optimizer_key: [ConditionedLocalAlgorithm, ...]}``.
    """
    traces_path = benchmark_dir(spec, scaling, variant, dim) / "traces.parquet"
    if not traces_path.exists():
        return None
    traces = load_traces_parquet(traces_path)
    family = family_for(spec, variant, dim)
    template = family.template
    optimum = problem_optimum(template)
    f0 = {
        seed: max(
            float(
                family.instance(seed).function(
                    family.instance(seed).starting_point(seed)
                )
            )
            - optimum,
            floor,
        )
        for seed in range(spec.num_seeds)
    }
    shifted = anchor_traces(gap_traces(traces, optimum, floor), f0)
    contenders = build_contenders(spec, variant, dim, scaling)
    return shifted, template, contenders


@app.function
def render_optimizer_figure(
    shifted,
    template,
    contenders: dict[str, list[ConditionedLocalAlgorithm]],
    optimizer_key: str,
    *,
    title: str,
    seed0: bool = False,
    floor: float = FLOOR,
) -> Figure | None:
    """One ``plot_convergence_overlay`` panel: all conditioners for a single
    optimizer, on one problem instance. Mirrors ``run_plot_stage``'s
    per-optimizer plotting inner loop, minus the ``save_path`` (so a live
    ``Figure`` comes back instead of a PNG on disk)."""
    runners = contenders.get(optimizer_key)
    if not runners:
        return None
    data = filter_seed(shifted, 0) if seed0 else shifted
    pooled = [
        t for runner in runners for t in data.get((template.name, runner.name), [])
    ]
    xmax = plot_xmax(pooled)
    return plot_convergence_overlay(
        data,
        template,
        runners,
        title=title,
        ylabel="$f(x_{best}) - f^*$",
        floor=floor,
        show_iqr=False,
        annotate_final=False,
        xmax=xmax,
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
    mo.md(
        "# Conditioning study browser\n"
        "Renders convergence overlays live from persisted `traces.parquet` "
        "files under `results/conditioning/exp1/`."
    )
    return


@app.cell
def _():
    studies = discover_studies(DATA_ROOT)
    by_name = group_by_name(studies)
    return by_name, studies


@app.cell(hide_code=True)
def _(mo, studies):
    mo.stop(
        not studies,
        mo.callout(
            mo.md(f"No studies found under `{DATA_ROOT}`. Run exp1 first."),
            kind="warn",
        ),
    )
    mo.md(f"Found {len(studies)} stud{'y' if len(studies) == 1 else 'ies'}.")
    return


# ---------------------------------------------------------------------------
# View 1: compare optimizers within one (name, rotate, scaling, variant, dim).
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md("## View 1 — compare optimizers")
    return


@app.cell(hide_code=True)
def _(mo, studies):
    mo.stop(not studies)

    study_options = {
        (
            f"{s.name} ({s.objective}) · rot={s.rotate}"
            if s.objective == ELLIPSOID_OBJECTIVE
            else f"{s.name} f{s.function_number:02d} ({s.objective}) · rot={s.rotate}"
        ): s
        for s in studies
    }
    study_sel = mo.ui.dropdown(
        list(study_options.keys()),
        value=next(iter(study_options), None),
        label="Study",
    )
    mo.vstack([study_sel])
    return study_options, study_sel


@app.cell(hide_code=True)
def _(mo, study_options, study_sel):
    mo.stop(study_sel.value is None)

    v1_study = study_options[study_sel.value]
    dim_sel = mo.ui.dropdown(
        [str(d) for d in v1_study.spec.dimensions],
        value=str(v1_study.spec.dimensions[0]),
        label="Dimension",
    )
    variant_sel = mo.ui.dropdown(
        list(v1_study.spec.variants),
        value=v1_study.spec.variants[0],
        label="Variant",
    )
    scaling_sel = mo.ui.dropdown(
        list(v1_study.scalings),
        value=v1_study.scalings[0] if v1_study.scalings else None,
        label="Scaling",
    )
    seed0_sel = mo.ui.checkbox(value=False, label="Seed 0 only (else pooled)")
    mo.hstack([dim_sel, variant_sel, scaling_sel, seed0_sel], gap=2)
    return dim_sel, scaling_sel, seed0_sel, v1_study, variant_sel


@app.cell(hide_code=True)
def _(dim_sel, mo, scaling_sel, seed0_sel, v1_study, variant_sel):
    mo.stop(
        dim_sel.value is None or variant_sel.value is None or scaling_sel.value is None,
        mo.callout(mo.md("No scaling data on disk for this study."), kind="warn"),
    )
    assert dim_sel.value is not None
    assert variant_sel.value is not None
    assert scaling_sel.value is not None

    v1_dim = int(dim_sel.value)
    v1_result = load_shifted(
        v1_study.spec, variant_sel.value, v1_dim, scaling_sel.value
    )
    mo.stop(
        v1_result is None,
        mo.callout(mo.md("No `traces.parquet` found for this selection."), kind="warn"),
    )
    assert v1_result is not None
    v1_shifted, v1_template, v1_contenders = v1_result

    v1_figs = []
    for v1_opt_key in v1_study.spec.optimizers:
        v1_title = (
            f"{v1_template.name}, d={v1_dim}, {variant_sel.value}, "
            f"{scaling_sel.value}, {LOCAL_LABELS.get(v1_opt_key, v1_opt_key)}"
        )
        v1_fig = render_optimizer_figure(
            v1_shifted,
            v1_template,
            v1_contenders,
            v1_opt_key,
            title=v1_title,
            seed0=seed0_sel.value,
        )
        if v1_fig is not None:
            v1_figs.append(mo.as_html(v1_fig))
            plt.close(v1_fig)

    mo.stop(not v1_figs, mo.callout(mo.md("No optimizer curves to show."), kind="warn"))
    v1_rows = [mo.hstack(v1_figs[i : i + 2], gap=1) for i in range(0, len(v1_figs), 2)]
    mo.vstack(v1_rows, gap=1)
    return


# ---------------------------------------------------------------------------
# View 2: sweep one variable (scaling / rotate / variant) for one optimizer.
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md("## View 2 — sweep scaling / rotate / variant")
    return


@app.cell(hide_code=True)
def _(by_name, mo):
    v2_names = sorted(by_name.keys())
    name_sel = mo.ui.dropdown(
        v2_names, value=v2_names[0] if v2_names else None, label="Study name"
    )
    mo.vstack([name_sel])
    return name_sel, v2_names


@app.cell(hide_code=True)
def _(by_name, mo, name_sel):
    mo.stop(name_sel.value is None)

    rot_map = by_name[name_sel.value]
    rotates_avail = sorted(rot_map.keys())
    ref_spec = rot_map[rotates_avail[0]].spec
    optimizer_sel = mo.ui.dropdown(
        list(ref_spec.optimizers),
        value=ref_spec.optimizers[0],
        label="Optimizer",
    )
    dim_sel2 = mo.ui.dropdown(
        [str(d) for d in ref_spec.dimensions],
        value=str(ref_spec.dimensions[0]),
        label="Dimension",
    )
    vary_sel = mo.ui.dropdown(
        ["scaling", "rotate", "variant"], value="scaling", label="Vary"
    )
    mo.hstack([optimizer_sel, dim_sel2, vary_sel], gap=2)
    return dim_sel2, optimizer_sel, rot_map, rotates_avail, vary_sel


@app.cell(hide_code=True)
def _(mo, rotates_avail):
    rotate_fixed_sel = mo.ui.dropdown(
        [str(r) for r in rotates_avail],
        value=str(rotates_avail[0]) if rotates_avail else None,
        label="Fixed rotate (ignored when vary = rotate)",
    )
    mo.vstack([rotate_fixed_sel])
    return (rotate_fixed_sel,)


@app.cell(hide_code=True)
def _(mo, rot_map, rotate_fixed_sel):
    mo.stop(rotate_fixed_sel.value is None)
    assert rotate_fixed_sel.value is not None

    ref_study = rot_map[int(rotate_fixed_sel.value)]
    variant_fixed_sel = mo.ui.dropdown(
        list(ref_study.spec.variants),
        value=ref_study.spec.variants[0],
        label="Fixed variant (ignored when vary = variant)",
    )
    scaling_fixed_sel = mo.ui.dropdown(
        list(ref_study.scalings),
        value=ref_study.scalings[0] if ref_study.scalings else None,
        label="Fixed scaling (ignored when vary = scaling)",
    )
    mo.hstack([variant_fixed_sel, scaling_fixed_sel], gap=2)
    return scaling_fixed_sel, variant_fixed_sel


@app.cell(hide_code=True)
def _(
    dim_sel2,
    mo,
    optimizer_sel,
    rot_map,
    rotate_fixed_sel,
    rotates_avail,
    scaling_fixed_sel,
    variant_fixed_sel,
    vary_sel,
):
    mo.stop(
        dim_sel2.value is None
        or optimizer_sel.value is None
        or rotate_fixed_sel.value is None,
        mo.callout(mo.md("Nothing to sweep for this study."), kind="warn"),
    )
    assert dim_sel2.value is not None
    assert optimizer_sel.value is not None
    assert rotate_fixed_sel.value is not None

    v2_dim = int(dim_sel2.value)
    v2_opt = optimizer_sel.value
    v2_vary = vary_sel.value
    v2_fixed_rotate = int(rotate_fixed_sel.value)

    if v2_vary == "rotate":
        v2_values: list = rotates_avail
    elif v2_vary == "scaling":
        v2_values = list(rot_map[v2_fixed_rotate].scalings)
    else:
        v2_values = list(rot_map[v2_fixed_rotate].spec.variants)

    v2_figs = []
    for v2_value in v2_values:
        v2_rotate = v2_value if v2_vary == "rotate" else v2_fixed_rotate
        v2_study = rot_map[v2_rotate]
        v2_variant = v2_value if v2_vary == "variant" else variant_fixed_sel.value
        v2_scaling = v2_value if v2_vary == "scaling" else scaling_fixed_sel.value
        if v2_variant is None or v2_scaling is None:
            continue
        v2_result = load_shifted(v2_study.spec, v2_variant, v2_dim, v2_scaling)
        if v2_result is None:
            continue
        v2_shifted, v2_template, v2_contenders = v2_result
        v2_title = (
            f"{v2_vary}={v2_value}  ·  {v2_template.name}, d={v2_dim}, "
            f"{v2_variant}, {v2_scaling}"
        )
        v2_fig = render_optimizer_figure(
            v2_shifted, v2_template, v2_contenders, v2_opt, title=v2_title
        )
        if v2_fig is not None:
            v2_figs.append(mo.as_html(v2_fig))
            plt.close(v2_fig)

    mo.stop(not v2_figs, mo.callout(mo.md("No data for this sweep."), kind="warn"))
    mo.hstack(v2_figs, gap=1)
    return


if __name__ == "__main__":
    app.run()
