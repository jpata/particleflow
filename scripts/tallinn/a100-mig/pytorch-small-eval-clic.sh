#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100-mig:1
#SBATCH --mem-per-gpu 100G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-12-03
cd ~/particleflow

WEIGHTS=/local/joosep/mlpf/results/clic/pyg-clic_20250106_193536_269746/checkpoints/checkpoint-05-1.995116.pth
singularity exec -B /scratch/persistent -B /local --nv \
     --env PYTHONPATH=`pwd` \
     --env KERAS_BACKEND=torch \
     $IMG python3 mlpf/pipeline.py --gpus 0 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
     --make-plots --gpu-batch-multiplier 100 --load $WEIGHTS --dtype bfloat16 --num-workers 0
