#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
require_arm_environment
cd "$EMPIREAI_REPO_ROOT"
exec python3 scripts/empireai/submit_scenario.py "$@"
