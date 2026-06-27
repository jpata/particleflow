#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 80G
#SBATCH --cpus-per-gpu 4
#SBATCH -o logs/slurm-%x-%a-%j-%N.out
#SBATCH --job-name=train-cld-hits-compare
#SBATCH --array=0-3

set -e
export PF_SITE=tallinn

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

nvidia-smi topo -m

MODELS=(attention gnnlsh hept heptv2)
MODEL_TYPE=${MODELS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=$(pixi run python3 scripts/get_param.py particleflow_spec.yaml productions.cld.workspace_dir)/tfds/
uv run python3 mlpf/pipeline.py \
    --spec-file particleflow_spec.yaml \
    --model-name pyg-cld-hits-v1 \
    --production cld \
    --prefix ${MODEL_TYPE}_ \
    --data-dir $DATA_DIR \
    train \
    --gpus 1 \
    --num_workers 4 \
    --prefetch_factor 2 \
    --model.type $MODEL_TYPE \
    --gpu_batch_multiplier 2 \
    --pad_to_multiple_elements 100
