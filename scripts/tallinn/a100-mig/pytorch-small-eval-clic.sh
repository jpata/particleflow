#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 200G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-18
cd ~/particleflow

WEIGHTS=experiments/pyg-clic_20241106_104416_929167/checkpoints/checkpoint-20-1.914489.pth
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=`pwd` \
     --env KERAS_BACKEND=torch \
     $IMG  python3 mlpf/pipeline.py --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
     --test --make-plots --gpu-batch-multiplier 100 --load $WEIGHTS --dtype bfloat16 --prefetch-factor 10 --num-workers 8 --ntest 50000
