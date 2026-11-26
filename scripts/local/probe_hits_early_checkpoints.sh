#!/bin/bash
# Extend the matched region-SSL probes at 20k and 40k, reusing completed controls.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PF_SITE=local
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TF_CPP_MIN_LOG_LEVEL=3
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mlpf-matplotlib}
comparison=experiments/cld_hits_ssl_comparison_5k_5seeds
mkdir -p "$comparison"
exec 9>"$comparison/run.lock"
flock -n 9 || { echo 'Another matched probe campaign is already running'; exit 1; }
trap 'printf "exit_code=%s finished=%s\n" "$?" "$(date --iso-8601=seconds)" > "$comparison/early_status.txt"' EXIT
printf 'running started=%s\n' "$(date --iso-8601=seconds)" > "$comparison/early_status.txt"
.venv/bin/python -m mlpf.probe_learning_curves \
  --run experiments/cld_hits_ssl_regions --output "$comparison/regions_early" \
  --ssl-steps 20000 40000 --seeds 31415 31416 31417 31418 31419 \
  --updates 5000 --parallel 2 --gpu-memory-fraction 0.46
.venv/bin/python "$comparison/ssl_step_benefit.py"
