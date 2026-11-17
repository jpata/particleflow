#!/bin/bash
# Same environment adapter contract as Flatiron and LUMI: prepare, then exec Python.
set -euo pipefail
REPO_ROOT=${MLPF_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$REPO_ROOT/scripts/empireai/environment.sh"
require_arm_environment
cd "$REPO_ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
exec python3 "$@"
