#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 60G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-07-08
cd ~/particleflow

env

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms-finetune.yaml \
    --train --test --make-plots --conv-type attention --attention-type flash --gpu-batch-multiplier 4 --num-workers 1 --prefetch-factor 50 --dtype bfloat16 --checkpoint-freq 1 --load experiments/pyg-cms_20240804_095032_809397_finetune/checkpoints/checkpoint-13-19.827532.pth --num-epochs 50
