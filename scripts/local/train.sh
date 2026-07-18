#!/bin/bash
set -euo pipefail

export PF_SITE=local

SPEC_FILE=${SPEC_FILE:-particleflow_spec.yaml}
USE_LOCAL_AVAILABLE_SPEC=${USE_LOCAL_AVAILABLE_SPEC:-true}
LOCAL_SPEC_FILE=${LOCAL_SPEC_FILE:-/tmp/particleflow_local_available_spec.yaml}
TARGETS=${TARGETS:-cld-hits}
SCENARIOS=${SCENARIOS:-shared,split}
DATA_CONFIG=${DATA_CONFIG:-1}

NUM_STEPS=${NUM_STEPS:-20000}
VAL_FREQ=${VAL_FREQ:-5000}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-5000}
GPU_BATCH_MULTIPLIER=${GPU_BATCH_MULTIPLIER:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
VALIDATION_DIAGNOSTICS_BATCHES=${VALIDATION_DIAGNOSTICS_BATCHES:-4}
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-experiments}

IFS=',' read -r -a TARGET_LIST <<< "$TARGETS"
IFS=',' read -r -a SCENARIO_LIST <<< "$SCENARIOS"

if [[ "$USE_LOCAL_AVAILABLE_SPEC" == "true" ]]; then
  uv run python3 scripts/local/make_local_available_spec.py "$SPEC_FILE" "$LOCAL_SPEC_FILE"
  SPEC_FILE="$LOCAL_SPEC_FILE"
fi

set_target() {
  local target=$1
  case "$target" in
    cld-hits)
      MODEL_NAME=pyg-cld-hits-v1
      PRODUCTION_NAME=cld
      ;;
    clic-hits)
      MODEL_NAME=pyg-clic-hits-v1
      PRODUCTION_NAME=clic
      ;;
    cld-pf)
      MODEL_NAME=pyg-cld-v1
      PRODUCTION_NAME=cld
      ;;
    clic-pf)
      MODEL_NAME=pyg-clic-v1
      PRODUCTION_NAME=clic
      ;;
    *)
      echo "Unknown target '$target'. Valid targets: cld-hits, clic-hits, cld-pf, clic-pf" >&2
      exit 1
      ;;
  esac
  DATA_DIR=$(python3 scripts/get_param.py "$SPEC_FILE" productions."$PRODUCTION_NAME".workspace_dir)/tfds/
}

make_common_args() {
  COMMON_ARGS=(
    --spec-file "$SPEC_FILE"
    --model-name "$MODEL_NAME"
    --production-name "$PRODUCTION_NAME"
    --data-dir "$DATA_DIR"
    --experiments-dir "$EXPERIMENTS_DIR"
    train
    --data_config "$DATA_CONFIG"
    --gpu_batch_multiplier "$GPU_BATCH_MULTIPLIER"
    --val_freq "$VAL_FREQ"
    --checkpoint_freq "$CHECKPOINT_FREQ"
    --num_steps "$NUM_STEPS"
    --num_workers "$NUM_WORKERS"
    --prefetch_factor "$PREFETCH_FACTOR"
    --sampler_mode interleaved-shards
    --validation_diagnostics_batches "$VALIDATION_DIAGNOSTICS_BATCHES"
    --make_plots
  )
}

run_scenario() {
  local target=$1
  local scenario=$2
  case "$scenario" in
    shared)
      uv run python3 mlpf/pipeline.py \
        --prefix "${target}_shared-task-query_" \
        "${COMMON_ARGS[@]}" \
        --model.backbone.mode shared \
        --model.backbone.num_convs 6 \
        --model.task_queries true
      ;;
    split)
      uv run python3 mlpf/pipeline.py \
        --prefix "${target}_split-backbone_" \
        "${COMMON_ARGS[@]}" \
        --model.backbone.mode split \
        --model.backbone.num_convs 3 \
        --model.task_queries false
      ;;
    *)
      echo "Unknown scenario '$scenario'. Valid scenarios: shared, split" >&2
      exit 1
      ;;
  esac
}

for target in "${TARGET_LIST[@]}"; do
  set_target "$target"
  make_common_args
  for scenario in "${SCENARIO_LIST[@]}"; do
    run_scenario "$target" "$scenario"
  done
done
