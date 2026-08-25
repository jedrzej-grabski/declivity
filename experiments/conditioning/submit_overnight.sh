#!/usr/bin/env bash
# Run exp1's and exp2's SLURM grids back-to-back, unattended.
#
# Not run in parallel: both scripts submit against the same account/QOS, and
# its per-user job caps (MaxJobsPU / MaxSubmitJobsPU, see
# conf/launcher/slurm.yaml) are pooled across everything you have queued, not
# per-script. Running them at the same time risks exceeding the submit cap
# and having one of the two `sbatch` calls rejected outright. Both scripts
# already block until their own SLURM array finishes, so running this
# wrapper sequentially is enough to keep them from ever overlapping.
#
# This process must stay alive for the whole run (it's just blocking on
# `sbatch`/polling, not doing the compute itself, but if it dies the
# in-flight submission's wait loop dies with it). Launch it under `nohup`,
# `tmux`, or `screen` so it survives your SSH session ending, e.g.:
#
#   tmux new -d -s conditioning 'experiments/conditioning/submit_overnight.sh'
#   tmux attach -t conditioning   # to check on it later
#
# or:
#
#   nohup experiments/conditioning/submit_overnight.sh \
#       > logs/submit_overnight_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Any trailing args are passed straight through as Hydra overrides to BOTH
# grids, same as calling either script directly.
set -euo pipefail
cd "$(dirname "$0")"

echo "[$(date)] Starting exp1 grid"
./exp1_submit_grid.sh "$@"
echo "[$(date)] exp1 grid done, starting exp2 grid"
./exp2_submit_grid.sh "$@"
echo "[$(date)] exp2 grid done"
