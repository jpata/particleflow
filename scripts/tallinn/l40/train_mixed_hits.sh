#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:2
#SBATCH --mem-per-gpu 80G
#SBATCH --cpus-per-gpu 4
#SBATCH -o logs/slurm-%x-%j-%N.out
#SBATCH --job-name=train-mixed-hits

set -euo pipefail
export PF_SITE=tallinn

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

nvidia-smi topo -m

SPEC_FILE=${SPEC_FILE:-particleflow_spec.yaml}
NUM_STEPS=${NUM_STEPS:-100000}
VAL_FREQ=${VAL_FREQ:-10000}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-10000}
GPU_BATCH_MULTIPLIER=${GPU_BATCH_MULTIPLIER:-24}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PAD_TO_MULTIPLE_ELEMENTS=${PAD_TO_MULTIPLE_ELEMENTS:-100}
VALIDATION_DIAGNOSTICS_BATCHES=${VALIDATION_DIAGNOSTICS_BATCHES:-8}
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-experiments}

CLD_DATA_DIR=$(pixi run python3 scripts/get_param.py "$SPEC_FILE" productions.cld.workspace_dir)/tfds
CLIC_DATA_DIR=$(pixi run python3 scripts/get_param.py "$SPEC_FILE" productions.clic.workspace_dir)/tfds
MIXED_DATA_DIR=${MIXED_DATA_DIR:-${TMPDIR:-/tmp}/particleflow_mixed_hits_tfds_${SLURM_JOB_ID}}
CLEAN_SPEC_FILE=${CLEAN_SPEC_FILE:-${TMPDIR:-/tmp}/particleflow_hit_training_spec_${SLURM_JOB_ID}.yaml}

mkdir -p "$MIXED_DATA_DIR" logs
ln -sfn "$CLD_DATA_DIR/cld_edm_ttbar_hits" "$MIXED_DATA_DIR/cld_edm_ttbar_hits"
ln -sfn "$CLD_DATA_DIR/cld_edm_qq_hits" "$MIXED_DATA_DIR/cld_edm_qq_hits"
ln -sfn "$CLD_DATA_DIR/cld_edm_ww_fullhad_hits" "$MIXED_DATA_DIR/cld_edm_ww_fullhad_hits"
ln -sfn "$CLIC_DATA_DIR/clic_edm_ttbar_hits" "$MIXED_DATA_DIR/clic_edm_ttbar_hits"
ln -sfn "$CLIC_DATA_DIR/clic_edm_qq_hits" "$MIXED_DATA_DIR/clic_edm_qq_hits"
ln -sfn "$CLIC_DATA_DIR/clic_edm_ww_fullhad_hits" "$MIXED_DATA_DIR/clic_edm_ww_fullhad_hits"
ls -al $MIXED_DATA_DIR

uv run python3 scripts/tallinn/l40/make_hit_training_spec.py "$SPEC_FILE" "$CLEAN_SPEC_FILE"

uv run python3 mlpf/pipeline.py \
    --spec-file "$CLEAN_SPEC_FILE" \
    --model-name mixed-hits \
    --production-name cld \
    --prefix "mixed_" \
    --data-dir "$MIXED_DATA_DIR" \
    --experiments-dir "$EXPERIMENTS_DIR" \
    train \
    --gpus 2 \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH_FACTOR" \
    --gpu_batch_multiplier "$GPU_BATCH_MULTIPLIER" \
    --num_steps "$NUM_STEPS" \
    --val_freq "$VAL_FREQ" \
    --checkpoint_freq "$CHECKPOINT_FREQ" \
    --pad_to_multiple_elements "$PAD_TO_MULTIPLE_ELEMENTS" \
    --sampler_mode interleaved-shards \
    --validation_diagnostics_batches "$VALIDATION_DIAGNOSTICS_BATCHES" \
    --test-datasets cld_edm_ttbar_hits clic_edm_ttbar_hits \
    --model.attention.use_jagged_attention True \
    --model.attention.use_flash_attn_varlen False \
    --model.backbone.mode shared \
    --model.backbone.num_convs 6 \
    --model.backbone.num_tracker_layers 2 \
    --model.backbone.num_calo_layers 2 \
    --model.backbone.num_common_layers 2 \
    --model.type attention \
    --model.task_queries false \
    --lr 0.0005
