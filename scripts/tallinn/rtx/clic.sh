#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:8
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-02
cd ~/particleflow

ulimit -n 10000
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3 mlpf/pyg_pipeline.py --dataset clic --gpus 8 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
    --train --conv-type attention --attention-type math --num-epochs 200 --gpu-batch-multiplier 32 --num-workers 4 --prefetch-factor 100 --checkpoint-freq 1 --comet
