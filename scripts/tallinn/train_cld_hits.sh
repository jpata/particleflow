#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:2
#SBATCH --mem-per-gpu 80G
#SBATCH --cpus-per-gpu 4
#SBATCH -o logs/slurm-%x-%j-%N.out
#SBATCH --job-name=train-cld-hits

set -e
export PF_SITE=tallinn

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

nvidia-smi topo -m

DATA_DIR=$(pixi run python3 scripts/get_param.py particleflow_spec.yaml productions.cld.workspace_dir)/tfds/
./scripts/tallinn/wrapper.sh python mlpf/pipeline.py \
    --spec-file particleflow_spec.yaml \
    --model-name pyg-cld-hits-v1 \
    --production cld \
    --data-dir $DATA_DIR \
    train \
    --gpus 2 \
    --lr 0.0005 \
    --lr_schedule onecycle \
    --num_workers 4 \
    --prefetch_factor 2 \
    --model.attention.num_convs 3 \
    --model.attention.head_dim 24
