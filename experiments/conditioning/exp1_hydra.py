"""Hydra entrypoint for :mod:`experiments.conditioning.exp1_conditioners`.

Config groups live under ``experiments/conditioning/conf/``:

    config.yaml       -- top-level defaults list + hydra.run/sweep.dir
    objective/         cec.yaml | ellipsoid.yaml
    experiment/         full.yaml | demo.yaml   (seed count, ks, budgets, ...)
    launcher/           local.yaml | slurm.yaml (submitit executor settings)

The sweep axes are the singular top-level fields ``dim``, ``variant``,
``rotate`` and ``function_number`` -- the facets that force a distinct CMA-ES
run (each lands in its own ``<name>/f{NN}/rot{0,1}/`` subtree; ``f{NN}`` is
omitted for ``objective=ellipsoid``, which has no per-function concept).
:func:`spec_from_cfg` lifts a single ``(dim, variant)`` pair into the
``Exp1Spec.dimensions``/``Exp1Spec.variants`` singleton tuples that the
pipeline expects. One SLURM array task = one ``(dim, variant, rotate,
function_number)`` cell, running the full setup -> cmaes -> hessian -> local
-> plot pipeline for that cell, with joblib parallelizing seeds across
``num_workers`` cores.

``scalings`` is deliberately NOT a sweep axis: scaling only reinterprets the
already-computed CMA-ES/Hessian matrices in the local stage, so one task
evaluates the whole ``scalings`` list against a single set of CMA-ES runs,
writing each into its own ``<name>/f{NN}/rot{0,1}/<scaling>/`` subtree.

Preview the composed config for one cell without running anything::

    PYTHONPATH=. uv run python -m experiments.conditioning.exp1_hydra \\
        --cfg job experiment=demo objective=ellipsoid dim=10 variant=bounded

Run a single cell locally (cheap smoke test)::

    PYTHONPATH=. uv run python -m experiments.conditioning.exp1_hydra \\
        experiment=demo objective=ellipsoid dim=10 variant=unbounded

Run the full 16-cell grid on this machine, one process per cell
(``hydra/launcher=submitit_local`` is the raw plugin default; our
``launcher=local`` wraps it with this suite's settings)::

    PYTHONPATH=. uv run python -m experiments.conditioning.exp1_hydra -m \\
        dim=10,30,50,100 variant=bounded,unbounded rotate=true,false \\
        experiment=full objective=cec launcher=local

Launch the same grid as a SLURM array (fill in the placeholders in
``conf/launcher/slurm.yaml`` -- partition/account/qos -- first)::

    PYTHONPATH=. uv run python -m experiments.conditioning.exp1_hydra -m \\
        dim=10,30,50,100 variant=bounded,unbounded rotate=true,false \\
        experiment=full objective=cec launcher=slurm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from experiments.conditioning.common import (
    BOUNDED,
    CEC_OBJECTIVE,
    ELLIPSOID_OBJECTIVE,
    SAMPLING_SPAN,
)
from experiments.conditioning.exp1_conditioners import Exp1Spec, run


@dataclass
class Exp1HydraConfig:
    """Flat, OmegaConf-friendly mirror of :class:`Exp1Spec`.

    ``Exp1Spec`` itself cannot be registered as a structured config: it has
    ``tuple``, ``Path``, and enum-valued fields that OmegaConf does not
    support. This schema restricts every field to plain ``int``/``float``/
    ``str``/``bool``/``list`` and :func:`spec_from_cfg` translates it into an
    ``Exp1Spec``.
    """

    # Objective (see conf/objective/*.yaml).
    objective: str = CEC_OBJECTIVE
    edition: str = "cec2017"
    function_number: int = 1

    # Sweep axes: one SLURM array task = one (dim, variant) cell.
    dim: int = 10
    variant: str = BOUNDED

    # Study identity/output roots.
    study_name: str | None = None
    data_root: str = "results/conditioning/exp1"
    plot_root: str = "plots/conditioning/exp1"

    # Experiment parameters (see conf/experiment/*.yaml).
    num_seeds: int = 25
    rotate: bool = True
    snapshot_ks: list[int] = field(default_factory=lambda: [2, 4, 8, 12, 16, 24, 32])
    cmaes_evaluations_per_dim: int = 10_000
    include_hessian: bool = True
    include_identity: bool = True
    optimizers: list[str] = field(
        default_factory=lambda: ["lbfgsb", "bfgs", "powell", "neldermead"]
    )
    transform: str = "inverse"
    # A study evaluates this whole list of scalings against one shared set of
    # CMA-ES runs (scaling only reinterprets the matrices in the local stage),
    # so scaling is an internal list, NOT a sweep axis.
    scalings: list[str] = field(default_factory=lambda: ["none"])
    population_factor: float | None = None
    sigma0: float = SAMPLING_SPAN / 5.0
    local_budget_per_dim: int = 500

    # Execution.
    num_workers: int = 1  # keep equal to launcher's cpus_per_task.
    replot: bool = False
    run_cmaes: bool = True
    force_cmaes: bool = False


cs = ConfigStore.instance()
cs.store(name="exp1_schema", node=Exp1HydraConfig)


def spec_from_cfg(cfg: Exp1HydraConfig) -> Exp1Spec:
    """Translate the composed Hydra config into an ``Exp1Spec`` for one
    ``(dim, variant)`` cell."""
    if cfg.objective == ELLIPSOID_OBJECTIVE:
        defaults = Exp1HydraConfig()
        if (
            cfg.edition != defaults.edition
            or cfg.function_number != defaults.function_number
        ):
            raise ValueError(
                "edition/function_number are not applicable with objective=ellipsoid."
            )
        edition = defaults.edition  # unused placeholder; ignored by family_for()
        function_number = defaults.function_number
        default_name = "ellipsoid"
    else:
        edition = cfg.edition
        function_number = cfg.function_number
        # function_number is its own path segment (study_root nests
        # f{NN}/rot{0,1} below the study name), so the default name doesn't
        # need to (and shouldn't, since it's the same across every function
        # in a sweep) encode it.
        default_name = edition

    return Exp1Spec(
        name=cfg.study_name if cfg.study_name is not None else default_name,
        objective=cfg.objective,
        edition=edition,
        function_number=function_number,
        dimensions=(cfg.dim,),
        num_seeds=cfg.num_seeds,
        variants=(cfg.variant,),
        rotate=cfg.rotate,
        snapshot_ks=tuple(cfg.snapshot_ks),
        cmaes_evaluations_per_dim=cfg.cmaes_evaluations_per_dim,
        include_hessian=cfg.include_hessian,
        include_identity=cfg.include_identity,
        optimizers=tuple(cfg.optimizers),
        transform=cfg.transform,
        scalings=tuple(cfg.scalings),
        population_factor=cfg.population_factor,
        sigma0=cfg.sigma0,
        local_budget_per_dim=cfg.local_budget_per_dim,
        num_workers=cfg.num_workers,
        data_root=Path(cfg.data_root),
        plot_root=Path(cfg.plot_root),
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Exp1HydraConfig) -> None:
    spec = spec_from_cfg(cfg)
    run(
        spec,
        replot=cfg.replot,
        run_cmaes=cfg.run_cmaes,
        force_cmaes=cfg.force_cmaes,
    )


if __name__ == "__main__":
    main()
