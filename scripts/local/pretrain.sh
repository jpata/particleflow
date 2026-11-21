#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PF_SITE=local
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TF_CPP_MIN_LOG_LEVEL=3
exec .venv/bin/python -m mlpf.pretrain \
  --config "${CONFIG:-configs/training/cld_ssl.yaml}" \
  --output "${OUTPUT:-experiments/cld_ssl_$(date +%Y%m%d_%H%M%S)}" "$@"
