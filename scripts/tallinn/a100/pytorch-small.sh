#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 60G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-07-08
cd ~/particleflow

env

# singularity exec -B /scratch/persistent --nv \
#     --env PYTHONPATH=hep_tfds \
#     --env KERAS_BACKEND=torch \
#     $IMG python3.10 mlpf/pyg_pipeline.py --dataset clic --gpus 1 \
#     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
#     --train --test --make-plots --conv-type attention --gpu-batch-multiplier 40 --num-workers 1 --prefetch-factor 50 --dtype bfloat16 --ntrain 10000 --ntest 10000 --nvalid 10000 --checkpoint-freq 1

# standalone evaluation
WEIGHTS=experiments/pyg-cms_20240710_123023_806687/checkpoints/checkpoint-09-19.719658.pth
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG  python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --load $WEIGHTS --ntest 10000
