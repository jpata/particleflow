#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 150G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-07-08
cd ~/particleflow

WEIGHTS=experiments/pyg-cms_20240804_095032_809397/checkpoints/checkpoint-16-19.681200.pth
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG  python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --gpu-batch-multiplier 20 --load $WEIGHTS --ntest 10000 --dtype bfloat16
