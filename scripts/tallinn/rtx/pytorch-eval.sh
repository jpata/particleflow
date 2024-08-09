#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu 60G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-02
cd ~/particleflow

env

WEIGHTS=experiments/pyg-cms_20240804_095032_809397/checkpoints/checkpoint-09-20.138613.pth
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG  python3 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --gpu-batch-multiplier 1 --attention-type math --load $WEIGHTS --ntest 10000 --dtype float16
