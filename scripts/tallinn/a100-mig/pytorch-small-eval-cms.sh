#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 150G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-18
cd ~/particleflow

WEIGHTS=experiments/pyg-cms_20240917_155112_841033/checkpoints/checkpoint-20-2.787848.pth
DATASET=$1
env
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=`pwd` \
     --env KERAS_BACKEND=torch \
     $IMG python mlpf/pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --gpu-batch-multiplier 2 --ntest 5000 --load $WEIGHTS --dtype bfloat16 --num-workers 8 --prefetch-factor 10 --load $WEIGHTS --test-datasets $DATASET
