#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2023-12-06
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --conv-type gnn_lsh --num-epochs 20 --gpu-batch-multiplier 4 --num-workers 1 --prefetch-factor 10 --ntrain 10000 --nvalid 10000
#    --train --conv-type mamba --num-epochs 20 --gpu-batch-multiplier 10 --num-workers 1 --prefetch-factor 10 --ntrain 10000 --nvalid 10000
