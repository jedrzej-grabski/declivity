#!/usr/bin/env bash
# Run the exp1 conditioning grid on this machine -- the local counterpart of
# exp1_submit_grid.sh.
#
# Same decomposition as the SLURM array: one cell per (dim, variant, rotate),
# each running the full setup -> cmaes -> hessian -> local -> plot pipeline and
# evaluating ALL scalings (from conf/experiment/full.yaml) against its single
# shared set of CMA-ES runs. With the defaults below that is
# 4 dims x 2 variants x 2 rotations = 16 cells.
#
# The difference is where the parallelism goes. On SLURM the cells run
# concurrently as array tasks. Here they run one at a time (launcher=serial,
# see conf/launcher/serial.yaml) and each cell parallelizes its seeds across
# WORKERS cores -- 16 concurrent cells would oversubscribe a single box.
#
# DIMS stays below 50: d=100 is a cluster-scale cell, and cell cost grows about
# quadratically in d (the local stage's budget is 500*d evaluations per run and
# each evaluation is O(d)).
#
# Usage:
#   experiments/conditioning/run_grid_local.sh                 # ellipsoid, full grid
#   OBJECTIVE=cec experiments/conditioning/run_grid_local.sh   # CEC F1 instead
#   DIMS=10,30 experiments/conditioning/run_grid_local.sh      # narrower sweep
#   WORKERS=6 experiments/conditioning/run_grid_local.sh       # leave cores free
#   experiments/conditioning/run_grid_local.sh num_seeds=10 experiment=demo
#     (any trailing args are passed straight through as Hydra overrides)
#
# Every stage is idempotent, so re-running after an interruption resumes rather
# than recomputing.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

OBJECTIVE=${OBJECTIVE:-ellipsoid}
# Ascending, so the cheap cells land results early and the cost of the
# expensive ones is already visible from the log by the time they start.
DIMS=${DIMS:-10,20,30,40}
VARIANTS=${VARIANTS:-bounded,unbounded}
ROTATE=${ROTATE:-true,false}
# Leave a couple of cores for the OS; joblib gets the rest.
WORKERS=${WORKERS:-$(( $(getconf _NPROCESSORS_ONLN) > 3 ? $(getconf _NPROCESSORS_ONLN) - 2 : 1 ))}

set -x
PYTHONPATH=. uv run python -m experiments.conditioning.exp1_hydra -m \
    dim="$DIMS" variant="$VARIANTS" rotate="$ROTATE" \
    experiment=full objective="$OBJECTIVE" launcher=serial \
    num_workers="$WORKERS" "$@"
