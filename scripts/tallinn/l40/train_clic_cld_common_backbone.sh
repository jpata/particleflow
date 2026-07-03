#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 80G
#SBATCH --cpus-per-gpu 4
#SBATCH -o logs/slurm-%x-%a-%j-%N.out
#SBATCH --job-name=train-clic-cld-common
#SBATCH --array=0-15

set -euo pipefail
export PF_SITE=tallinn

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

nvidia-smi topo -m

SPEC_FILE=${SPEC_FILE:-particleflow_spec.yaml}
NUM_STEPS=${NUM_STEPS:-20000}
VAL_FREQ=${VAL_FREQ:-5000}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-5000}
GPU_BATCH_MULTIPLIER=${GPU_BATCH_MULTIPLIER:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PAD_TO_MULTIPLE_ELEMENTS=${PAD_TO_MULTIPLE_ELEMENTS:-100}
DATA_CONFIG=${DATA_CONFIG:-1}

DATASETS=(mixed cld)
TARGETS=(hits pf)
SCENARIOS=(baseline stems modality modality-source)

NUM_TARGETS=${#TARGETS[@]}
NUM_SCENARIOS=${#SCENARIOS[@]}
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_TARGETS * NUM_SCENARIOS)))
TARGET_INDEX=$(((SLURM_ARRAY_TASK_ID / NUM_SCENARIOS) % NUM_TARGETS))
SCENARIO_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SCENARIOS))

DATASET_SCOPE=${DATASETS[$DATASET_INDEX]}
TARGET=${TARGETS[$TARGET_INDEX]}
SCENARIO=${SCENARIOS[$SCENARIO_INDEX]}

CLD_DATA_DIR=$(pixi run python3 scripts/get_param.py "$SPEC_FILE" productions.cld.workspace_dir)/tfds
CLIC_DATA_DIR=$(pixi run python3 scripts/get_param.py "$SPEC_FILE" productions.clic.workspace_dir)/tfds
MIXED_DATA_DIR=${MIXED_DATA_DIR:-${TMPDIR:-/tmp}/particleflow_clic_cld_common_tfds_${SLURM_JOB_ID}}
COMMON_SPEC_FILE=${COMMON_SPEC_FILE:-${TMPDIR:-/tmp}/particleflow_clic_cld_common_spec_${SLURM_JOB_ID}.yaml}

mkdir -p "$MIXED_DATA_DIR" logs
ln -sfn "$CLD_DATA_DIR/cld_edm_ttbar_hits" "$MIXED_DATA_DIR/cld_edm_ttbar_hits"
ln -sfn "$CLD_DATA_DIR/cld_edm_ttbar_pf" "$MIXED_DATA_DIR/cld_edm_ttbar_pf"
ln -sfn "$CLIC_DATA_DIR/clic_edm_ttbar_hits" "$MIXED_DATA_DIR/clic_edm_ttbar_hits"
ln -sfn "$CLIC_DATA_DIR/clic_edm_ttbar_pf" "$MIXED_DATA_DIR/clic_edm_ttbar_pf"

uv run python3 scripts/tallinn/l40/make_clic_cld_common_spec.py "$SPEC_FILE" "$COMMON_SPEC_FILE"

if [[ "$DATASET_SCOPE" == "mixed" && "$TARGET" == "hits" ]]; then
    MODEL_NAME=pyg-clic-cld-hits-v1
    DATA_DIR="$MIXED_DATA_DIR"
elif [[ "$DATASET_SCOPE" == "mixed" && "$TARGET" == "pf" ]]; then
    MODEL_NAME=pyg-clic-cld-v1
    DATA_DIR="$MIXED_DATA_DIR"
elif [[ "$DATASET_SCOPE" == "cld" && "$TARGET" == "hits" ]]; then
    MODEL_NAME=pyg-cld-hits-v1
    DATA_DIR="$CLD_DATA_DIR"
elif [[ "$DATASET_SCOPE" == "cld" && "$TARGET" == "pf" ]]; then
    MODEL_NAME=pyg-cld-v1
    DATA_DIR="$CLD_DATA_DIR"
else
    echo "Unsupported dataset/target combination: ${DATASET_SCOPE}/${TARGET}" >&2
    exit 1
fi

COMMON_ARGS=(
    --spec-file "$COMMON_SPEC_FILE"
    --model-name "$MODEL_NAME"
    --production-name cld
    --prefix "${DATASET_SCOPE}-${TARGET}-${SCENARIO}_"
    --data-dir "$DATA_DIR"
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
)

case "$SCENARIO" in
    baseline)
        uv run python3 mlpf/pipeline.py \
            "${COMMON_ARGS[@]}" \
            --model.backbone.mode shared \
            --model.backbone.num_convs 6 \
            --model.task_queries true \
            --model.input_stem.mode standard
        ;;
    stems)
        uv run python3 mlpf/pipeline.py \
            "${COMMON_ARGS[@]}" \
            --model.backbone.mode shared \
            --model.backbone.num_convs 6 \
            --model.task_queries true \
            --model.input_stem.mode modality \
            --model.input_stem.modality_embedding false \
            --model.input_stem.source_embedding false \
            --model.input_stem.input_norm true
        ;;
    modality)
        uv run python3 mlpf/pipeline.py \
            "${COMMON_ARGS[@]}" \
            --model.backbone.mode shared \
            --model.backbone.num_convs 6 \
            --model.task_queries true \
            --model.input_stem.mode modality \
            --model.input_stem.modality_embedding true \
            --model.input_stem.source_embedding false \
            --model.input_stem.input_norm true
        ;;
    modality-source)
        uv run python3 mlpf/pipeline.py \
            "${COMMON_ARGS[@]}" \
            --model.backbone.mode shared \
            --model.backbone.num_convs 6 \
            --model.task_queries true \
            --model.input_stem.mode modality \
            --model.input_stem.modality_embedding true \
            --model.input_stem.source_embedding true \
            --model.input_stem.input_norm true
        ;;
    *)
        echo "Unknown scenario: $SCENARIO" >&2
        exit 1
        ;;
esac
