#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-$REPO_ROOT/particleflow-env/bin/python3}
if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "Python environment not found at '$PYTHON_EXECUTABLE'; set PYTHON_EXECUTABLE" >&2
  exit 2
fi

exec "$PYTHON_EXECUTABLE" scripts/lumi/submit_scenario.py "$@"
