#!/usr/bin/env bash
# Submit the exp2 hybrid grid to SLURM via Hydra + hydra-submitit-launcher.
#
# Two rounds, because exp2 has a suite-wide ECDF aggregated over every
# function, which a single-function compute cell cannot produce:
#
#   1. Heavy compute: one SLURM array task per (dim, function_number) cell,
#      running setup -> cmaes -> probes -> compose (plus a throwaway
#      single-function plot) for that cell. With the defaults below that is
#      4 dims x 30 functions = 120 cells.
#   2. Aggregate: one lightweight, LOCAL (not submitted) task per dim that
#      reruns just the plot stage with the full function list, producing the
#      real suite-wide ECDF from every function's persisted Parquet.
#
# Round 1 is submitted ONE DIM AT A TIME (30 array elements per submission),
# not as one 120-element array: QOS plgrid1 caps this account at
# MaxSubmitJobsPU=40 (running + pending combined, array elements counted
# individually the instant they're submitted -- see conf/launcher/slurm.yaml),
# so a single 120-element array would be rejected outright.
# hydra-submitit-launcher blocks until each dim's array finishes before the
# loop submits the next one, which also keeps you under that cap with
# headroom to spare. Round 2 only starts once every dim's compute has
# finished, since it reads round 1's output from disk.
#
# Cluster settings (partition/account/qos/resources) live in
# conf/launcher/slurm.yaml -- edit there, not here.
#
# Usage:
#   experiments/conditioning/exp2_submit_grid.sh                 # full grid
#   DIMS=10,30 experiments/conditioning/exp2_submit_grid.sh      # narrower sweep
#   experiments/conditioning/exp2_submit_grid.sh num_seeds=10 ks='[0.5,1,2,4,8]'
#     (any trailing args are passed straight through as Hydra overrides, to
#      BOTH rounds)
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

DIMS=${DIMS:-10,30,50,100}
FUNCTIONS=${FUNCTIONS:-$(seq -s, 1 30)}

set -x
IFS=',' read -ra dim_list <<< "$DIMS"
for dim in "${dim_list[@]}"; do
    PYTHONPATH=. uv run python -m experiments.conditioning.exp2_hydra -m \
        dim="$dim" function_number="$FUNCTIONS" \
        experiment2=full launcher=slurm "$@"
done

PYTHONPATH=. uv run python -m experiments.conditioning.exp2_hydra -m \
    dim="$DIMS" \
    experiment2=full launcher=local replot=true "$@"
