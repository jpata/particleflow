#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 100G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-12-03
cd ~/particleflow

WEIGHTS=experiments/pyg-cms_20250729_112213_357186/checkpoints/checkpoint-10-3.812354.pth
EXPDIR=experiments/pyg-cms_20250729_112213_357186

DATASET=$1
env
singularity exec -B /local -B /scratch/persistent --nv \
     --env PYTHONPATH=`pwd` \
     --env KERAS_BACKEND=torch \
     $IMG python mlpf/pipeline.py \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets \
     --config parameters/pytorch/pyg-cms.yaml \
     --experiment-dir $EXPDIR \
     test \
     --make-plots \
     --gpus 1 \
     --gpu-batch-multiplier 2 \
     --load $WEIGHTS \
     --ntest 5000 \
     --dtype bfloat16 \
     --num-workers 1 \
     --prefetch-factor 10 \
     --test-datasets $DATASET
