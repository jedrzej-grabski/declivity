# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

"""Interactive browser for exp1 conditioning-study results.

Displays the pre-rendered PNGs under ``plots/conditioning/exp1/`` (written by
``exp1_conditioners.py::run_plot_stage``) instead of re-rendering
``traces.parquet`` on every parameter change -- swapping study/dim/variant/
scaling is instant instead of waiting on a live re-plot. See ``visualize.py``
for the live-rendering counterpart; this file mirrors its UI and study
discovery, just swaps `load_shifted`/`render_optimizer_figure` for a path
lookup into the plot tree.

Launch with:

    PYTHONPATH=. uv run marimo edit experiments/conditioning/visualize_static.py
"""

from __future__ import annotations

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    # Ensure the project root is on the path regardless of cwd.
    _ROOT = Path(__file__).parent.parent.parent.resolve()
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from experiments.conditioning.common import ELLIPSOID_OBJECTIVE, LOCAL_LABELS
    from experiments.conditioning.exp1_conditioners import Exp1Spec
    from experiments.conditioning.visualize import (
        DATA_ROOT,
        discover_studies,
        group_by_name,
    )

    PLOT_ROOT = Path("plots/conditioning/exp1")


@app.function
def plot_dir_for(spec: Exp1Spec, scaling: str, variant: str, dim: int) -> Path:
    """Mirrors the ``out`` directory ``run_plot_stage`` writes PNGs into."""
    rot = f"rot{int(spec.rotate)}"
    return PLOT_ROOT / spec.name / rot / scaling / variant / f"d{dim:03d}"


@app.function
def png_path(
    spec: Exp1Spec,
    scaling: str,
    variant: str,
    dim: int,
    optimizer_key: str,
    seed0: bool = False,
) -> Path:
    suffix = "_convergence_seed0.png" if seed0 else "_convergence.png"
    return plot_dir_for(spec, scaling, variant, dim) / f"{optimizer_key}{suffix}"


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        "# Conditioning study browser (static)\n"
        "Displays pre-rendered PNGs from `plots/conditioning/exp1/`. Run "
        "`exp1_hydra.py`'s plot stage first to (re)generate them for any "
        "selection that shows as missing below."
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
    v1_images = []
    for v1_opt_key in v1_study.spec.optimizers:
        v1_path = png_path(
            v1_study.spec,
            scaling_sel.value,
            variant_sel.value,
            v1_dim,
            v1_opt_key,
            seed0=seed0_sel.value,
        )
        if v1_path.exists():
            v1_images.append(
                mo.vstack(
                    [
                        mo.md(f"**{LOCAL_LABELS.get(v1_opt_key, v1_opt_key)}**"),
                        mo.image(v1_path),
                    ]
                )
            )

    mo.stop(
        not v1_images,
        mo.callout(
            mo.md(
                f"No pre-rendered PNGs found under "
                f"`{plot_dir_for(v1_study.spec, scaling_sel.value, variant_sel.value, v1_dim)}`. "
                "Run the plot stage for this selection first."
            ),
            kind="warn",
        ),
    )
    v1_rows = [
        mo.hstack(v1_images[i : i + 3], gap=1, widths="equal")
        for i in range(0, len(v1_images), 3)
    ]
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
    return (name_sel,)


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

    v2_images = []
    for v2_value in v2_values:
        v2_rotate = v2_value if v2_vary == "rotate" else v2_fixed_rotate
        v2_study = rot_map[v2_rotate]
        v2_variant = v2_value if v2_vary == "variant" else variant_fixed_sel.value
        v2_scaling = v2_value if v2_vary == "scaling" else scaling_fixed_sel.value
        if v2_variant is None or v2_scaling is None:
            continue
        v2_path = png_path(v2_study.spec, v2_scaling, v2_variant, v2_dim, v2_opt)
        if not v2_path.exists():
            continue
        v2_images.append(
            mo.vstack([mo.md(f"**{v2_vary}={v2_value}**"), mo.image(v2_path)])
        )

    mo.stop(not v2_images, mo.callout(mo.md("No data for this sweep."), kind="warn"))
    mo.vstack(v2_images, gap=1, align="center")
    return


if __name__ == "__main__":
    app.run()
