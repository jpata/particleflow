#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 80G
#SBATCH --cpus-per-gpu 4
#SBATCH -o logs/slurm-%x-%j-%N.out
#SBATCH --job-name=train-clic

set -e
export PF_SITE=tallinn

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

nvidia-smi topo -m

DATA_DIR=$(pixi run python3 scripts/get_param.py particleflow_spec.yaml productions.clic.workspace_dir)/tfds/
./scripts/tallinn/wrapper.sh python mlpf/pipeline.py \
    --spec-file particleflow_spec.yaml \
    --model-name pyg-clic-v1 \
    --production clic \
    --data-dir $DATA_DIR \
    train \
    --gpus 1 \
    --num_workers 4 \
    --prefetch_factor 2
