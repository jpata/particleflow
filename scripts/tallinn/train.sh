#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:2
#SBATCH --mem-per-gpu 40G
#SBATCH --cpus-per-gpu 4
#SBATCH -o logs/slurm-%x-%j-%N.out
set -e
export PF_SITE=tallinn

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

nvidia-smi topo -m

DATA_DIR=$(pixi run python3 scripts/get_param.py particleflow_spec.yaml productions.cld.workspace_dir)/tfds/
./scripts/tallinn/wrapper.sh python mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-cld-hits-v1 --production cld --data-dir $DATA_DIR train --gpus 2 --num_workers 4 --prefetch_factor 2 --load experiments/pyg-cld-hits-v1_cld_20260325_102238_932912/checkpoints/checkpoint-40000.pth
