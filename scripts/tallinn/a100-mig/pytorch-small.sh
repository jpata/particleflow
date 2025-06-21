#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 60G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2025-06-12
cd ~/particleflow2

env

ulimit -n 10000
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=`pwd` \
    --env KERAS_BACKEND=torch \
    $IMG python3 mlpf/pipeline.py --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --test --make-plots --conv-type litemla --gpu-batch-multiplier 5 --num-workers 1 --prefetch-factor 10 --dtype bfloat16 --checkpoint-freq -1 --ntrain 10000 --nvalid 1000 --ntest 1000 --num-epochs 10
