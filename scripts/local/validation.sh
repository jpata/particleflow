#!/bin/bash
set -euo pipefail

export PF_SITE=local
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/particleflow-matplotlib}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SPEC_FILE=${SPEC_FILE:-particleflow_spec.yaml}
LOCAL_SPEC_FILE=${LOCAL_SPEC_FILE:-/tmp/particleflow_local_validation_spec.yaml}
CHECKPOINT=${CHECKPOINT:-experiments/pyg-cld-hits-v1_cld_20260814_013544_620216/checkpoints/checkpoint-20000.pth}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/pyg-cld-hits-v1_cld_20260814_013544_620216/validation_checkpoint_20000_n5000_with_pf}
NTEST=${NTEST:-5000}
GPU_BATCH_MULTIPLIER=${GPU_BATCH_MULTIPLIER:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
# The default checkpoint predates the neighborhood summaries and was trained
# with 12 raw columns plus the original 15 geometry features.
HIT_INPUT_DIM=${HIT_INPUT_DIM:-12}
HIT_FEATURE_ENGINEERING=${HIT_FEATURE_ENGINEERING:-true}
HIT_GEOMETRY_FEATURES=${HIT_GEOMETRY_FEATURES:-true}
HIT_TRACKER_NEIGHBORHOOD_FEATURES=${HIT_TRACKER_NEIGHBORHOOD_FEATURES:-false}
HIT_CALORIMETER_NEIGHBORHOOD_FEATURES=${HIT_CALORIMETER_NEIGHBORHOOD_FEATURES:-false}

CHECKPOINT=$(realpath "$CHECKPOINT")
OUTPUT_ROOT=$(realpath -m "$OUTPUT_ROOT")
TRAINING_HISTORY_DIR=${TRAINING_HISTORY_DIR:-$(dirname "$(dirname "$CHECKPOINT")")/history}

# The local TFDS installation contains split 1 for both hit-based datasets.
# Generate a matching spec so pipeline.py does not try to open unavailable splits.
uv run python3 scripts/local/make_local_available_spec.py "$SPEC_FILE" "$LOCAL_SPEC_FILE"
SPEC_FILE="$LOCAL_SPEC_FILE"

CLD_DATA_DIR=$(python3 scripts/get_param.py "$SPEC_FILE" productions.cld.workspace_dir)/tfds
CLIC_DATA_DIR=$(python3 scripts/get_param.py "$SPEC_FILE" productions.clic.workspace_dir)/tfds

test -f "$CHECKPOINT"
test -d "$CLD_DATA_DIR/cld_edm_ttbar_hits"
test -d "$CLIC_DATA_DIR/clic_edm_ttbar_hits"
mkdir -p "$OUTPUT_ROOT/cld" "$OUTPUT_ROOT/clic"

run_validation() {
  local model_name=$1
  local production_name=$2
  local data_dir=$3
  local sample=$4
  local output_dir=$5

  uv run python3 -u mlpf/pipeline.py \
    --spec-file "$SPEC_FILE" \
    --model-name "$model_name" \
    --production-name "$production_name" \
    --data-dir "$data_dir" \
    --experiment-dir "$output_dir" \
    test \
    --gpus 1 \
    --load "$CHECKPOINT" \
    --test-datasets "$sample" \
    --ntest "$NTEST" \
    --make-plots \
    --dtype bfloat16 \
    --gpu_batch_multiplier "$GPU_BATCH_MULTIPLIER" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH_FACTOR" \
    --pad_to_multiple_elements 100 \
    --input_dim "$HIT_INPUT_DIM" \
    --model.type attention \
    --model.hit_feature_engineering.enabled "$HIT_FEATURE_ENGINEERING" \
    --model.hit_feature_engineering.geometry "$HIT_GEOMETRY_FEATURES" \
    --model.hit_feature_engineering.tracker_neighborhood "$HIT_TRACKER_NEIGHBORHOOD_FEATURES" \
    --model.hit_feature_engineering.calorimeter_neighborhood "$HIT_CALORIMETER_NEIGHBORHOOD_FEATURES" \
    --model.backbone.mode shared \
    --model.backbone.num_convs 6 \
    --model.attention.num_convs 6 \
    --model.attention.use_jagged_attention true \
    --model.attention.use_flash_attn_varlen false \
    --model.task_queries false
}

# run_validation \
#   pyg-cld-hits-v1 cld "$CLD_DATA_DIR" cld_edm_ttbar_hits "$OUTPUT_ROOT/cld"

uv run python3 -u scripts/local/plot_hit_validation_with_pf.py \
  --validation-dir "$OUTPUT_ROOT/cld" \
  --hit-sample cld_edm_ttbar_hits \
  --pf-data-dir "$CLD_DATA_DIR" \
  --pf-sample cld_edm_ttbar_pf \
  --dataset cld_hits \
  --num-events "$NTEST"

uv run python3 -u scripts/local/plot_training_evolution.py \
  --history-dir "$TRAINING_HISTORY_DIR" \
  --output-dir "$OUTPUT_ROOT/cld/plots_test/cld_edm_ttbar_hits" \
  --sample cld_edm_ttbar_hits

# run_validation \
#   pyg-clic-hits-v1 clic "$CLIC_DATA_DIR" clic_edm_ttbar_hits "$OUTPUT_ROOT/clic"

uv run python3 -u scripts/local/plot_hit_validation_with_pf.py \
  --validation-dir "$OUTPUT_ROOT/clic" \
  --hit-sample clic_edm_ttbar_hits \
  --pf-data-dir "$CLIC_DATA_DIR" \
  --pf-sample clic_edm_ttbar_pf \
  --dataset clic_hits \
  --num-events "$NTEST"

echo "Validation predictions and plot_utils.py plots written under: $OUTPUT_ROOT"
