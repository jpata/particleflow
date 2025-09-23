#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 400G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2025-09-01
cd ~/particleflow

ulimit -n 100000
singularity exec -B /scratch/persistent -B /local --nv \
    --env PYTHONPATH=`pwd` \
    --env KERAS_BACKEND=torch \
    $IMG python3 mlpf/pipeline.py \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets \
    --config parameters/pytorch/pyg-cms.yaml \
    train \
    --gpus 1 \
    --gpu-batch-multiplier 16 \
    --num-workers 2 \
    --prefetch-factor 2 \
    --conv-type attention \
    --dtype bfloat16 \
    --optimizer adamw \
    --comet \
    --test-datasets cms_pf_qcd \
    --num-steps 100000
