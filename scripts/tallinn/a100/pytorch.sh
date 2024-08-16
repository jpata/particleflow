#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 200G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-02
cd ~/particleflow

ulimit -n 10000
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms-ttbar-nopu.yaml \
    --train --test --make-plots --num-epochs 20 --conv-type attention --gpu-batch-multiplier 10 --num-workers 4 --prefetch-factor 100 --checkpoint-freq 1 --comet --ntrain 50000 --nvalid 5000 --ntest 5000
