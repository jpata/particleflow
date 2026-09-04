#!/bin/bash
set -euo pipefail

export PF_SITE=local

SPEC_FILE=${SPEC_FILE:-particleflow_spec.yaml}
USE_LOCAL_AVAILABLE_SPEC=${USE_LOCAL_AVAILABLE_SPEC:-true}
LOCAL_SPEC_FILE=${LOCAL_SPEC_FILE:-/tmp/particleflow_local_ttbar_comparison_spec.yaml}
OUTPUT_MODES=${OUTPUT_MODES:-elementwise,set}
HIT_VERSION=${HIT_VERSION:-3.2.1}
HIT_SPLITS=${HIT_SPLITS:-1}
DATA_CONFIG=${DATA_CONFIG:-${HIT_SPLITS// /,}}

NUM_STEPS=${NUM_STEPS:-2000}
VAL_FREQ=${VAL_FREQ:-200}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-200}
NVALID=${NVALID:-100}
NTEST=${NTEST:-100}
GPU_BATCH_MULTIPLIER=${GPU_BATCH_MULTIPLIER:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
VALIDATION_DIAGNOSTICS_BATCHES=${VALIDATION_DIAGNOSTICS_BATCHES:-4}
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-experiments}
PAD_TO_MULTIPLE_ELEMENTS=${PAD_TO_MULTIPLE_ELEMENTS:-128}

IFS=',' read -r -a OUTPUT_MODE_LIST <<< "$OUTPUT_MODES"
read -r -a HIT_SPLIT_LIST <<< "$HIT_SPLITS"

if [[ "$USE_LOCAL_AVAILABLE_SPEC" == "true" ]]; then
  uv run python3 scripts/local/make_local_available_spec.py \
    "$SPEC_FILE" "$LOCAL_SPEC_FILE" \
    --hit-version "$HIT_VERSION" \
    --hit-splits "${HIT_SPLIT_LIST[@]}"
  SPEC_FILE="$LOCAL_SPEC_FILE"
fi

PRODUCTION_NAME=cld
DATA_DIR=${DATA_DIR:-$(uv run python3 scripts/get_param.py "$SPEC_FILE" productions."$PRODUCTION_NAME".workspace_dir)/tfds/}

set_output_mode() {
  local output_mode=$1
  case "$output_mode" in
    elementwise)
      MODEL_NAME=pyg-cld-hits-v1
      ;;
    set)
      MODEL_NAME=pyg-cld-hits-set-v1
      ;;
    *)
      echo "Unknown output mode '$output_mode'. Valid modes: elementwise, set" >&2
      exit 1
      ;;
  esac
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
    --nvalid "$NVALID"
    --ntest "$NTEST"
    --num_workers "$NUM_WORKERS"
    --prefetch_factor "$PREFETCH_FACTOR"
    --sampler_mode interleaved-shards
    --validation_diagnostics_batches "$VALIDATION_DIAGNOSTICS_BATCHES"
    --make_plots
    --pad_to_multiple_elements "$PAD_TO_MULTIPLE_ELEMENTS"
  )
}

run_comparison_training() {
  local output_mode=$1
  echo "Starting CLD ttbar hit training with output_mode=$output_mode model=$MODEL_NAME"
  uv run python3 mlpf/pipeline.py \
    --prefix "ttbar-${output_mode}_" \
    "${COMMON_ARGS[@]}"
}

for output_mode in "${OUTPUT_MODE_LIST[@]}"; do
  set_output_mode "$output_mode"
  make_common_args
  run_comparison_training "$output_mode"
done
