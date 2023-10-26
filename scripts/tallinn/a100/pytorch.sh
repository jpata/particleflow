#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 0 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pyg-cms.yaml \
    --train --conv-type gnn_lsh --num-epochs 10 --gpu-batch-multiplier 10 --num-workers 1 --prefetch-factor 10 --ntrain 200
