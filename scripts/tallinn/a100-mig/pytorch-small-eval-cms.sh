#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 100G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-18
cd ~/particleflow

WEIGHTS=experiments/pyg-cms_20241101_090645_682892/checkpoints/checkpoint-08-2.986092.pth
DATASET=$1
env
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=`pwd` \
     --env KERAS_BACKEND=torch \
     $IMG python mlpf/pipeline.py --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms-nopu.yaml \
     --test --make-plots --gpu-batch-multiplier 2 --load $WEIGHTS --ntest 20000 --dtype bfloat16 --num-workers 8 --prefetch-factor 10 --test-datasets $DATASET
