#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 250G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-12-03
cd ~/particleflow

ulimit -n 100000
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=`pwd` \
    --env KERAS_BACKEND=torch \
    $IMG python3 mlpf/pipeline.py --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
    --train --conv-type attention \
    --gpu-batch-multiplier 256 --checkpoint-freq 1 --num-workers 8 --prefetch-factor 100 --comet --ntest 2000 --test-datasets clic_edm_qq_pf
