#!/bin/bash
#SBATCH --job-name=mlpf_evolution
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mig:6
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH -o logs/evolution-%x-%j.out
#SBATCH -e logs/evolution-%x-%j.err

set -e
set -o xtrace

export PF_SITE=tallinn
export PYTHONUNBUFFERED=1

# Ensure log directory exists
mkdir -p logs/evolution

# Dataset directory from the previous context
DATA_DIR="/local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/tfds/"

env
nvidia-smi

stdbuf -oL -eL apptainer exec --nv -B /local -B /cvmfs -B /scratch/local -B /scratch/persistent --env PYTHONPATH=/home/joosep/particleflow2 /scratch/persistent/joosep/singularity/pytorch-20260305-08d6950.sif \
    python3 mlpf/standalone/run_evolution.py \
    --data-dir $DATA_DIR \
    --generations 20 \
    --pop-size 100 \
    --log-dir logs/evolution \
    -v
