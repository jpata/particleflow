#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:4
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-18

ulimit -n 10000
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3 mlpf/pyg_pipeline.py --dataset clic --gpus 4 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
    --train --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --num-workers 2 --prefetch-factor 100 --attention-type math --dtype float32
