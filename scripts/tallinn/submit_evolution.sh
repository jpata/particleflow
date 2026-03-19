#!/bin/bash
#SBATCH --job-name=mlpf_evolution
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mig:6
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o logs/evolution-%x-%j.out
#SBATCH -e logs/evolution-%x-%j.err

set -e
set -o xtrace

export PF_SITE=tallinn

# Ensure log directory exists
mkdir -p logs/evolution

# Dataset directory from the previous context
DATA_DIR="/local/joosep/mlpf/cms/20260204_cmssw_15_0_5_117d32/tfds/"

# Use the tallinn wrapper to run the evolution script
# We specify --generations and --pop-size as arguments
# The script will automatically detect the 6 MIG GPUs
./scripts/tallinn/wrapper.sh python3 mlpf/standalone/run_evolution.py \
    --data-dir $DATA_DIR \
    --generations 10 \
    --pop-size 100 \
    --log-dir logs/evolution \
    -v
