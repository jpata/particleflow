#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 80G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-02-13
cd ~/particleflow

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --conv-type attention --num-epochs 10 --gpu-batch-multiplier 30 --num-workers 2 --prefetch-factor 20 --checkpoint-freq 1 --comet

# singularity exec -B /scratch/persistent --nv \
#     --env PYTHONPATH=hep_tfds \
#     $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
#     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
#     --test --make-plots --conv-type attention --gpu-batch-multiplier 40 --num-workers 2 --prefetch-factor 20 --load experiments/pyg-cms_20240218_204141_378871/checkpoints/checkpoint-05-48.080534.pth --ntest 1000
