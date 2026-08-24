"""Hydra entrypoint for :mod:`experiments.conditioning.exp2_hybrid`.

Config groups live under ``experiments/conditioning/conf/``:

    config2.yaml       -- top-level defaults list + hydra.run/sweep.dir
    experiment2/         full.yaml | demo.yaml   (seed count, ks, budgets, ...)
    launcher/            local.yaml | slurm.yaml (submitit executor settings,
                          shared with exp1_hydra)

CEC 2017 only (exp2_hybrid has no ellipsoid path), so there is no
``objective`` group.

The sweep axis is the singular top-level field ``dim`` -- CMA-ES/probe cost
scales steeply with dimension, so one SLURM array task = one dim, running
the full setup -> cmaes -> probes -> compose -> plot pipeline for every
function in ``functions`` and every scaling in ``scalings``, with joblib
parallelizing (function, seed, optimizer) jobs across ``num_workers`` cores.
:func:`spec_from_cfg` lifts the singular ``dim`` into ``Exp2Spec.dimensions``'
singleton tuple that the pipeline expects.

``scalings`` is deliberately NOT split per array task: scaling only
reinterprets the already-computed CMA-ES snapshots in the probe stage, so one
task evaluates the whole ``scalings`` list against a single set of CMA-ES
runs, writing each into its own ``<name>/<scaling>/`` subtree (see
``Exp2Spec.scalings`` / ``scaling_root`` in ``exp2_hybrid.py``).

Preview the composed config for one cell without running anything::

    PYTHONPATH=. uv run python -m experiments.conditioning.exp2_hydra \\
        --cfg job experiment2=demo dim=10

Run a single cell locally (cheap smoke test)::

    PYTHONPATH=. uv run python -m experiments.conditioning.exp2_hydra \\
        experiment2=demo dim=10

Run the full grid on this machine, one process per cell
(``hydra/launcher=submitit_local`` is the raw plugin default; our
``launcher=local`` wraps it with this suite's settings)::

    PYTHONPATH=. uv run python -m experiments.conditioning.exp2_hydra -m \\
        dim=10,30,50,100 experiment2=full launcher=local

Launch the same grid as a SLURM array (fill in the placeholders in
``conf/launcher/slurm.yaml`` -- partition/account/qos -- first)::

    PYTHONPATH=. uv run python -m experiments.conditioning.exp2_hydra -m \\
        dim=10,30,50,100 experiment2=full launcher=slurm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from experiments.conditioning.common import BOUNDED, SAMPLING_SPAN
from experiments.conditioning.exp2_hybrid import (
    DEFAULT_EVALUATIONS_PER_DIMENSION,
    Exp2Spec,
    run,
)


@dataclass
class Exp2HydraConfig:
    """Flat, OmegaConf-friendly mirror of :class:`Exp2Spec`.

    ``Exp2Spec`` itself cannot be registered as a structured config: it has
    ``tuple``/``Path``-valued fields that OmegaConf does not support. This
    schema restricts every field to plain ``int``/``float``/``str``/``bool``/
    ``list`` and :func:`spec_from_cfg` translates it into an ``Exp2Spec``.
    """

    edition: str = "cec2017"
    functions: list[int] = field(default_factory=lambda: list(range(1, 31)))

    # Sweep axis: one SLURM array task = one dim.
    dim: int = 10

    # Study identity/output roots.
    study_name: str | None = None
    data_root: str = "results/conditioning/exp2"
    plot_root: str = "plots/conditioning/exp2"

    # Experiment parameters (see conf/experiment2/*.yaml).
    num_seeds: int = 25
    variant: str = BOUNDED
    rotate: bool = False
    ks: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0])
    granularity: float = 0.5
    cmaes_evaluations_per_dim: int = DEFAULT_EVALUATIONS_PER_DIMENSION
    population_factor: float = 4.0
    sigma0: float = SAMPLING_SPAN / 5.0
    probe_budget_per_dim: int = 200
    optimizers: list[str] = field(
        default_factory=lambda: [
            "lbfgsb",
            "bfgs",
            "powell",
            "neldermead_control",
            "neldermead",
            "neldermead_hc",
            "neldermead_hc_shaped",
        ]
    )
    transform: str = "inverse"
    # A study evaluates this whole list of scalings against one shared set of
    # CMA-ES runs (scaling only reinterprets the snapshots in the probe
    # stage), so scaling is an internal list, NOT a sweep axis.
    scalings: list[str] = field(default_factory=lambda: ["none"])

    # Execution.
    num_workers: int = 1  # keep equal to launcher's cpus_per_task.
    replot: bool = False
    run_cmaes: bool = True
    force_cmaes: bool = False
    force_probes: bool = False


cs = ConfigStore.instance()
cs.store(name="exp2_schema", node=Exp2HydraConfig)


def spec_from_cfg(cfg: Exp2HydraConfig) -> Exp2Spec:
    """Translate the composed Hydra config into an ``Exp2Spec`` for one
    ``dim`` cell."""
    return Exp2Spec(
        name=cfg.study_name if cfg.study_name is not None else cfg.edition,
        edition=cfg.edition,
        functions=tuple(cfg.functions),
        dimensions=(cfg.dim,),
        num_seeds=cfg.num_seeds,
        variant=cfg.variant,
        rotate=cfg.rotate,
        ks=tuple(cfg.ks),
        granularity=cfg.granularity,
        cmaes_evaluations_per_dim=cfg.cmaes_evaluations_per_dim,
        population_factor=cfg.population_factor,
        sigma0=cfg.sigma0,
        probe_budget_per_dim=cfg.probe_budget_per_dim,
        optimizers=tuple(cfg.optimizers),
        transform=cfg.transform,
        scalings=tuple(cfg.scalings),
        num_workers=cfg.num_workers,
        data_root=Path(cfg.data_root),
        plot_root=Path(cfg.plot_root),
    )


@hydra.main(version_base=None, config_path="conf", config_name="config2")
def main(cfg: Exp2HydraConfig) -> None:
    spec = spec_from_cfg(cfg)
    run(
        spec,
        replot=cfg.replot,
        run_cmaes=cfg.run_cmaes,
        force_cmaes=cfg.force_cmaes,
        force_probes=cfg.force_probes,
    )


if __name__ == "__main__":
    main()
