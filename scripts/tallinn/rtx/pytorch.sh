#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:2
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 0,1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pyg-cms-small.yaml \
    --train --conv-type gnn_lsh --num-epochs 10 --ntrain 500 --ntest 500 --gpu-batch-multiplier 1 --num-workers 1 --prefetch-factor 10
