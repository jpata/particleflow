#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 80G
#SBATCH --cpus-per-gpu 4
#SBATCH -o logs/slurm-%x-%a-%j-%N.out
#SBATCH --job-name=train-hit-cluster-long
#SBATCH --array=0-5

set -euo pipefail
export PF_SITE=tallinn

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

nvidia-smi topo -m

SPEC_FILE=${SPEC_FILE:-particleflow_spec.yaml}
NUM_STEPS=${NUM_STEPS:-50000}
VAL_FREQ=${VAL_FREQ:-10000}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-10000}
GPU_BATCH_MULTIPLIER=${GPU_BATCH_MULTIPLIER:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PAD_TO_MULTIPLE_ELEMENTS=${PAD_TO_MULTIPLE_ELEMENTS:-100}
VALIDATION_DIAGNOSTICS_BATCHES=${VALIDATION_DIAGNOSTICS_BATCHES:-8}
CLUSTERING_LOSS_WEIGHT_MEDIUM=${CLUSTERING_LOSS_WEIGHT_MEDIUM:-0.10}
CLUSTERING_LOSS_WEIGHT_STRONG=${CLUSTERING_LOSS_WEIGHT_STRONG:-0.30}
DATA_CONFIG=${DATA_CONFIG:-1}
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-experiments}

# Long hit-only clustering confirmation matrix:
#   H0: hit particle-number clustering does not improve hit-task FOM.
#   H1: the best clustering weights from the short sweep improve the learned hit
#       representation and increase per-dataset jet matching fraction over the
#       no-clustering baseline for CLD-only, CLIC-only, and mixed CLIC+CLD hits.
# Fixed architecture:
#   shared backbone, modality-specific hit stems, no modality embedding.
# Success criterion:
#   each clustering run should beat the matched no-clustering baseline on the
#   average of CLD and CLIC hit-test jet matching fractions at 50k steps.
JOBS=(
    cld-hits:stems
    cld-hits:stems-cluster010
    clic-hits:stems
    clic-hits:stems-cluster030
    mixed-hits:stems
    mixed-hits:stems-cluster030
)

IFS=: read -r TRAINSET SCENARIO <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

CLD_DATA_DIR=$(pixi run python3 scripts/get_param.py "$SPEC_FILE" productions.cld.workspace_dir)/tfds
CLIC_DATA_DIR=$(pixi run python3 scripts/get_param.py "$SPEC_FILE" productions.clic.workspace_dir)/tfds
MIXED_DATA_DIR=${MIXED_DATA_DIR:-${TMPDIR:-/tmp}/particleflow_hit_fom_tfds_${SLURM_JOB_ID}}
CLEAN_SPEC_FILE=${CLEAN_SPEC_FILE:-${TMPDIR:-/tmp}/particleflow_hit_fom_clean_spec_${SLURM_JOB_ID}.yaml}

mkdir -p "$MIXED_DATA_DIR" logs
ln -sfn "$CLD_DATA_DIR/cld_edm_ttbar_hits" "$MIXED_DATA_DIR/cld_edm_ttbar_hits"
ln -sfn "$CLIC_DATA_DIR/clic_edm_ttbar_hits" "$MIXED_DATA_DIR/clic_edm_ttbar_hits"

uv run python3 scripts/tallinn/l40/make_hit_fom_clean_spec.py "$SPEC_FILE" "$CLEAN_SPEC_FILE"

case "$TRAINSET" in
    cld-hits)
        MODEL_NAME=pyg-clean-cld-hits-v1
        ;;
    clic-hits)
        MODEL_NAME=pyg-clean-clic-hits-v1
        ;;
    mixed-hits)
        MODEL_NAME=pyg-clean-mixed-hits-v1
        ;;
    *)
        echo "Unknown trainset: $TRAINSET" >&2
        exit 1
        ;;
esac

COMMON_ARGS=(
    --spec-file "$CLEAN_SPEC_FILE"
    --model-name "$MODEL_NAME"
    --production-name cld
    --prefix "hit-fom-${TRAINSET}-${SCENARIO}_"
    --data-dir "$MIXED_DATA_DIR"
    --experiments-dir "$EXPERIMENTS_DIR"
    train
    --gpus 1
    --num_workers "$NUM_WORKERS"
    --prefetch_factor "$PREFETCH_FACTOR"
    --gpu_batch_multiplier "$GPU_BATCH_MULTIPLIER"
    --num_steps "$NUM_STEPS"
    --val_freq "$VAL_FREQ"
    --checkpoint_freq "$CHECKPOINT_FREQ"
    --data_config "$DATA_CONFIG"
    --pad_to_multiple_elements "$PAD_TO_MULTIPLE_ELEMENTS"
    --sampler_mode interleaved-shards
    --validation_diagnostics_batches "$VALIDATION_DIAGNOSTICS_BATCHES"
    --test-datasets cld_edm_ttbar_hits clic_edm_ttbar_hits
)

MODEL_ARGS=(
    --model.backbone.mode shared
    --model.backbone.num_convs 6
    --model.task_queries true
    --model.input_stem.mode modality
    --model.input_stem.source_embedding false
    --model.input_stem.input_norm true
)

case "$SCENARIO" in
    stems)
        MODEL_ARGS+=(--model.input_stem.modality_embedding false)
        ;;
    stems-cluster010)
        MODEL_ARGS+=(--model.input_stem.modality_embedding false --clustering_loss.weight "$CLUSTERING_LOSS_WEIGHT_MEDIUM")
        ;;
    stems-cluster030)
        MODEL_ARGS+=(--model.input_stem.modality_embedding false --clustering_loss.weight "$CLUSTERING_LOSS_WEIGHT_STRONG")
        ;;
    *)
        echo "Unknown scenario: $SCENARIO" >&2
        exit 1
        ;;
esac

uv run python3 mlpf/pipeline.py \
    "${COMMON_ARGS[@]}" \
    "${MODEL_ARGS[@]}"
