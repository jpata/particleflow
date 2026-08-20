#!/bin/bash
set -euo pipefail

export PF_SITE=local

SPEC_FILE=${SPEC_FILE:-particleflow_spec.yaml}
USE_LOCAL_AVAILABLE_SPEC=${USE_LOCAL_AVAILABLE_SPEC:-true}
LOCAL_SPEC_FILE=${LOCAL_SPEC_FILE:-/tmp/particleflow_local_available_spec.yaml}
TARGETS=${TARGETS:-cld-hits}
DATA_CONFIG=${DATA_CONFIG:-1}

NUM_STEPS=${NUM_STEPS:-2000}
VAL_FREQ=${VAL_FREQ:-1000}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-1000}
GPU_BATCH_MULTIPLIER=${GPU_BATCH_MULTIPLIER:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
VALIDATION_DIAGNOSTICS_BATCHES=${VALIDATION_DIAGNOSTICS_BATCHES:-4}
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-experiments}
NUM_TRACKER_LAYERS=${NUM_TRACKER_LAYERS:-2}
NUM_CALO_LAYERS=${NUM_CALO_LAYERS:-2}
NUM_COMMON_LAYERS=${NUM_COMMON_LAYERS:-2}

IFS=',' read -r -a TARGET_LIST <<< "$TARGETS"

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
    --model.attention.use_jagged_attention false
    --pad_to_multiple_elements 100
  )
}

run_detector_scenario() {
  local target=$1
  if [[ "$target" != "cld-hits" && "$target" != "clic-hits" ]]; then
    echo "Detector-specific training is only valid for cld-hits and clic-hits" >&2
    exit 1
  fi
  local num_detector_layers=$((NUM_TRACKER_LAYERS + NUM_CALO_LAYERS + NUM_COMMON_LAYERS))
  uv run python3 mlpf/pipeline.py \
    --prefix "${target}_detector-backbone_" \
    "${COMMON_ARGS[@]}" \
    --model.backbone.mode shared \
    --model.backbone.num_convs "$num_detector_layers" \
    --model.backbone.num_tracker_layers "$NUM_TRACKER_LAYERS" \
    --model.backbone.num_calo_layers "$NUM_CALO_LAYERS" \
    --model.backbone.num_common_layers "$NUM_COMMON_LAYERS" \
    --model.attention.use_jagged_attention true \
    --model.task_queries false
}

for target in "${TARGET_LIST[@]}"; do
  set_target "$target"
  make_common_args
  run_detector_scenario "$target"
done
