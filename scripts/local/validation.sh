#!/usr/bin/env bash
set -euo pipefail

export PF_SITE=local
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/particleflow-matplotlib}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

spec_file=${SPEC_FILE:-particleflow_spec.yaml}
local_spec_file=${LOCAL_SPEC_FILE:-/tmp/particleflow_local_physics_validation_spec.yaml}
output_root=${OUTPUT_ROOT:-notebooks/studies/20260819_hit_vs_pf_comparison/physics_validation}
ntest=${NTEST:-1000}
gpu_batch_multiplier=${GPU_BATCH_MULTIPLIER:-4}
num_workers=${NUM_WORKERS:-4}
prefetch_factor=${PREFETCH_FACTOR:-2}
run_hit=${RUN_HIT:-true}
run_pf=${RUN_PF:-true}

hit_checkpoint=${HIT_CHECKPOINT:-experiments/pyg-cld-hits-v1_cld_20260815_011516_887054/checkpoints/checkpoint-20000.pth}
pf_checkpoint=${PF_CHECKPOINT:-experiments/pyg-cld-v1_cld_20260815_135053_268848/checkpoints/checkpoint-40000.pth}
hit_data_dir=${HIT_DATA_DIR:-data/tfds_validation_cld/tensorflow_datasets/cld}
pf_data_dir=${PF_DATA_DIR:-data/tfds_validation_cld/tensorflow_datasets/cld}
hit_version=${HIT_VERSION:-3.2.0}
hit_split=${HIT_SPLIT:-1}
pf_version=${PF_VERSION:-3.2.0}
pf_split=${PF_SPLIT:-1}

hit_checkpoint=$(realpath "$hit_checkpoint")
pf_checkpoint=$(realpath "$pf_checkpoint")
hit_data_dir=$(realpath "$hit_data_dir")
pf_data_dir=$(realpath "$pf_data_dir")
output_root=$(realpath -m "$output_root")

test -f "$hit_checkpoint"
test -f "$pf_checkpoint"
test -d "$hit_data_dir/cld_edm_ttbar_hits/$hit_split/$hit_version"
test -d "$pf_data_dir/cld_edm_ttbar_pf/$pf_split/$pf_version"
mkdir -p "$output_root/hit" "$output_root/pf"

uv run python3 scripts/local/make_local_available_spec.py \
  "$spec_file" "$local_spec_file" \
  --hit-version "$hit_version" --hit-splits "$hit_split" \
  --pf-version "$pf_version" --pf-splits "$pf_split"

common_args=(--spec-file "$local_spec_file" --production-name cld)
test_args=(
  test --gpus 1 --ntest "$ntest" --make-plots
  --dtype bfloat16 --gpu_batch_multiplier "$gpu_batch_multiplier"
  --num_workers "$num_workers" --prefetch_factor "$prefetch_factor"
  --pad_to_multiple_elements 100
  --model.type attention --model.backbone.mode shared
  --model.backbone.num_convs 6 --model.attention.num_convs 3
  --model.attention.use_jagged_attention true
  --model.attention.use_flash_attn_varlen false
  --model.task_queries false
)

if [[ "$run_hit" == true ]]; then
  echo "Running hit-input physics validation ($ntest events)"
  uv run python3 -u mlpf/pipeline.py \
    "${common_args[@]}" --model-name pyg-cld-hits-v1 \
    --data-dir "$hit_data_dir" --experiment-dir "$output_root/hit" \
    "${test_args[@]}" --load "$hit_checkpoint" \
    --test-datasets cld_edm_ttbar_hits --input_dim 15 \
    --model.hit_feature_engineering.enabled true \
    --model.hit_feature_engineering.geometry true \
    --model.hit_feature_engineering.tracker_neighborhood false \
    --model.hit_feature_engineering.calorimeter_neighborhood true \
    --model.backbone.num_tracker_layers 2 \
    --model.backbone.num_calo_layers 2 \
    --model.backbone.num_common_layers 2
fi

if [[ "$run_pf" == true ]]; then
  echo "Running track/cluster-input physics validation ($ntest events)"
  uv run python3 -u mlpf/pipeline.py \
    "${common_args[@]}" --model-name pyg-cld-v1 \
    --data-dir "$pf_data_dir" --experiment-dir "$output_root/pf" \
    "${test_args[@]}" --load "$pf_checkpoint" \
    --test-datasets cld_edm_ttbar_pf --input_dim 17
fi

echo "Physics validation outputs written under: $output_root"
