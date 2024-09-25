#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 200G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-18
cd ~/particleflow

WEIGHTS=experiments/pyg-cms_20240917_155112_841033/checkpoints/checkpoint-18-2.810102.pth
env
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG python mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --gpu-batch-multiplier 2 --load $WEIGHTS --ntest 50000 --dtype bfloat16 --num-workers 8 --prefetch-factor 10
