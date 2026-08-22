#!/usr/bin/env bash
# Submit the exp1 conditioning grid to SLURM via Hydra + hydra-submitit-launcher.
#
# One SLURM array task per (dim, variant, rotate) cell; each task runs the full
# setup -> cmaes -> hessian -> local -> plot pipeline and evaluates ALL scalings
# (from conf/experiment/full.yaml) against its single shared set of CMA-ES runs.
# With the defaults below that is 4 dims x 2 variants x 2 rotations = 16 tasks.
#
# Cluster settings (partition/account/qos/resources) live in
# conf/launcher/slurm.yaml -- edit there, not here.
#
# Usage:
#   experiments/conditioning/submit_grid.sh                 # ellipsoid, full grid
#   OBJECTIVE=cec experiments/conditioning/submit_grid.sh   # CEC F1 instead
#   DIMS=10,30 experiments/conditioning/submit_grid.sh      # narrower sweep
#   experiments/conditioning/submit_grid.sh num_seeds=10 snapshot_ks='[2,4,8,16,32,64]'
#     (any trailing args are passed straight through as Hydra overrides)
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

OBJECTIVE=${OBJECTIVE:-ellipsoid}
DIMS=${DIMS:-10,30,50,100}
VARIANTS=${VARIANTS:-bounded,unbounded}
ROTATE=${ROTATE:-true,false}

set -x
PYTHONPATH=. uv run python -m experiments.conditioning.exp1_hydra -m \
    dim="$DIMS" variant="$VARIANTS" rotate="$ROTATE" \
    experiment=full objective="$OBJECTIVE" launcher=slurm "$@"
