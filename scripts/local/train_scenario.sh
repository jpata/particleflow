#!/bin/bash
# Pick and run a reusable training scenario on the local platform.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
SCENARIO_DIR="$REPO_ROOT/configs/training/scenarios"

print_choices() {
  echo "Scenarios:"
  local scenario
  for scenario in "$SCENARIO_DIR"/*.yaml; do
    [[ -e "$scenario" ]] || continue
    basename "$scenario" .yaml
  done
}

if [[ ${1:-} == "--list" ]]; then
  print_choices
  exit 0
fi
if [[ $# -eq 0 ]]; then
  print_choices
  echo "Usage: $0 SCENARIO [RUNNER_OPTIONS...]" >&2
  exit 2
fi

SCENARIO_REFERENCE=$1
shift
if [[ -f "$SCENARIO_REFERENCE" ]]; then
  SCENARIO_FILE=$(cd "$(dirname "$SCENARIO_REFERENCE")" && pwd)/$(basename "$SCENARIO_REFERENCE")
else
  SCENARIO_NAME=${SCENARIO_REFERENCE%.yaml}
  SCENARIO_FILE="$SCENARIO_DIR/$SCENARIO_NAME.yaml"
fi
if [[ ! -f "$SCENARIO_FILE" ]]; then
  echo "Unknown training scenario: $SCENARIO_REFERENCE" >&2
  print_choices >&2
  exit 2
fi

cd "$REPO_ROOT"
export PF_SITE=local

PLATFORM_FILE=${PLATFORM_FILE:-configs/training/platforms/local.yaml}
SPEC_FILE=${SPEC_FILE:-$(uv run python3 scripts/get_param.py "$SCENARIO_FILE" spec_file particleflow_spec.yaml)}
PRODUCTION_NAME=$(uv run python3 scripts/get_param.py "$SCENARIO_FILE" production_name)
USE_LOCAL_AVAILABLE_SPEC=${USE_LOCAL_AVAILABLE_SPEC:-true}
LOCAL_SPEC_FILE=${LOCAL_SPEC_FILE:-/tmp/particleflow_local_available_spec.yaml}
SEED=${SEED:-}
HIT_VERSION=${HIT_VERSION:-3.2.1}
HIT_SPLITS=${HIT_SPLITS:-1}
DATA_CONFIG=${DATA_CONFIG:-${HIT_SPLITS// /,}}

# Local defaults intentionally shorten the generic comparison scenario. Every
# value remains overridable through the existing environment interface or by
# supplying a later runner option on this command line.
NUM_STEPS=${NUM_STEPS:-2000}
VAL_FREQ=${VAL_FREQ:-200}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-200}
NVALID=${NVALID:-100}
NTEST=${NTEST:-100}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${GPU_BATCH_MULTIPLIER:-8}}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
VALIDATION_DIAGNOSTICS_BATCHES=${VALIDATION_DIAGNOSTICS_BATCHES:-4}
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-experiments}
PAD_TO_MULTIPLE_ELEMENTS=${PAD_TO_MULTIPLE_ELEMENTS:-128}

read -r -a HIT_SPLIT_LIST <<< "$HIT_SPLITS"
if [[ "$USE_LOCAL_AVAILABLE_SPEC" == "true" ]]; then
  uv run python3 scripts/local/make_local_available_spec.py \
    "$SPEC_FILE" "$LOCAL_SPEC_FILE" \
    --hit-version "$HIT_VERSION" \
    --hit-splits "${HIT_SPLIT_LIST[@]}"
  SPEC_FILE="$LOCAL_SPEC_FILE"
fi

DATA_DIR=${DATA_DIR:-$(uv run python3 scripts/get_param.py "$SPEC_FILE" productions."$PRODUCTION_NAME".workspace_dir)/tfds/}

RUN_ARGS=(
  --scenario "$SCENARIO_FILE"
  --platform "$PLATFORM_FILE"
  --spec-file "$SPEC_FILE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --data-dir "$DATA_DIR"
  --experiments-dir "$EXPERIMENTS_DIR"
  --set "data_config=$DATA_CONFIG"
  --set "num_steps=$NUM_STEPS"
  --set "val_freq=$VAL_FREQ"
  --set "checkpoint_freq=$CHECKPOINT_FREQ"
  --set "nvalid=$NVALID"
  --set "ntest=$NTEST"
  --set "num_workers=$NUM_WORKERS"
  --set "prefetch_factor=$PREFETCH_FACTOR"
  --set "validation_diagnostics_batches=$VALIDATION_DIAGNOSTICS_BATCHES"
  --set "pad_to_multiple_elements=$PAD_TO_MULTIPLE_ELEMENTS"
)
if [[ -n "$SEED" ]]; then
  RUN_ARGS+=(--seed "$SEED")
fi

exec uv run python3 scripts/training/run_scenario.py "${RUN_ARGS[@]}" "$@"
