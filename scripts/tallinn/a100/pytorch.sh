#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 300G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-12-03
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
    --gpu-batch-multiplier 10 \
    --checkpoint-freq 1 \
    --num-workers 8 \
    --prefetch-factor 10 \
    --comet \
    --ntrain 5000 \
    --nvalid 1000 \
    --ntest 1000 \
    --test-datasets cms_pf_qcd --load experiments/pyg-cms_20250729_100004_087561/checkpoints/checkpoint-09-3.818757.pth
