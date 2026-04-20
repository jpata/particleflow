#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem 80G
#SBATCH --cpus-per-task 8
#SBATCH -o logs/slurm-%x-%j-%N.out
#SBATCH --job-name=onnx

set -e
export PF_SITE=tallinn

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

nvidia-smi topo -m

uv run snakemake --snakefile scripts/local/Snakefile_model_validation --cores 1
