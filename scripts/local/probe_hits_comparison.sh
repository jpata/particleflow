#!/bin/bash
# Paired five-seed comparison at matched SSL event exposure. Safe to restart.
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
flock -n 9 || { echo 'This comparison is already running'; exit 1; }
trap 'printf "exit_code=%s finished=%s\n" "$?" "$(date --iso-8601=seconds)" > "$comparison/status.txt"' EXIT
printf 'running started=%s\n' "$(date --iso-8601=seconds)" > "$comparison/status.txt"
common=(--updates 5000 --seeds 31415 31416 31417 31418 31419 --parallel 2 --gpu-memory-fraction 0.46)
.venv/bin/python -m mlpf.probe_learning_curves \
  --run experiments/cld_hits_ssl_regions --output "$comparison/regions" \
  --ssl-steps 0 60000 80000 100000 "${common[@]}"
.venv/bin/python -m mlpf.probe_learning_curves \
  --run experiments/cld_hits_ssl_large --output "$comparison/thinning" \
  --ssl-steps 0 30000 40000 50000 "${common[@]}"
.venv/bin/python "$comparison/compare.py"
