#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CONFIG=${CONFIG:-configs/training/cld_hits_ssl.yaml}
export OUTPUT=${OUTPUT:-experiments/cld_hits_ssl_$(date +%Y%m%d_%H%M%S)}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec bash scripts/local/pretrain.sh "$@"
