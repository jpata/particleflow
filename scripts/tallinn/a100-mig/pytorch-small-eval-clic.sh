#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100-mig:1
#SBATCH --mem-per-gpu 100G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-12-03
cd ~/particleflow

WEIGHTS=experiments/pyg-clic_20250106_193536_269746/checkpoints/checkpoint-05-1.995116.pth
_EXPDIR=experiments/pyg-clic_20250106_193536_269746
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=`pwd` \
     --env KERAS_BACKEND=torch \
     $IMG  python3 mlpf/pipeline.py --gpus 0 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
     --make-plots --gpu-batch-multiplier 100 --load $WEIGHTS --dtype bfloat16 --num-workers 0 --experiment-dir $_EXPDIR
