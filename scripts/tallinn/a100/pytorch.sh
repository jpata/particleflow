#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 80G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-03-11
cd ~/particleflow

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --conv-type attention --num-epochs 100 --gpu-batch-multiplier 20 --num-workers 4 --prefetch-factor 50 --checkpoint-freq 1 --comet --load experiments/pyg-cms_20240331_100326_706273/checkpoints/checkpoint-02-18.142171.pth

# singularity exec -B /scratch/persistent --nv \
#     --env PYTHONPATH=hep_tfds \
#     $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
#     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
#     --test --make-plots --conv-type attention --gpu-batch-multiplier 40 --num-workers 2 --prefetch-factor 20 --load experiments/pyg-cms_20240218_204141_378871/checkpoints/checkpoint-05-48.080534.pth --ntest 1000
