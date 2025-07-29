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
    $IMG python3 mlpf/pipeline.py --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --conv-type attention \
    --gpu-batch-multiplier 10 --checkpoint-freq 1 --num-workers 4 --prefetch-factor 10 --comet --ntest 1000 --test-datasets cms_pf_qcd \
    --num-epochs 12 --load experiments/pyg-cms_20250722_101813_274478/checkpoints/checkpoint-10-3.812332.pth --lr-schedule constant
