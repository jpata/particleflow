#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 100G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-12-03
cd ~/particleflow

WEIGHTS=experiments/pyg-clic_20250306_105311_290722/checkpoints/checkpoint-10-1.934364.pth
_EXPDIR=experiments/pyg-clic_20250306_105311_290722/
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=`pwd` \
     --env KERAS_BACKEND=torch \
     $IMG  python3 mlpf/pipeline.py --gpus 0 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
     --make-plots --ntest 5000 --gpu-batch-multiplier 100 --load $WEIGHTS --dtype bfloat16 --num-workers 0 --experiment-dir $_EXPDIR
