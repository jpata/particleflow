#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 80G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-02-05
cd ~/particleflow

#pytorch training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --conv-type attention --num-epochs 20 --gpu-batch-multiplier 20 --num-workers 2 --prefetch-factor 20
#    --train --conv-type gnn_lsh --num-epochs 20 --gpu-batch-multiplier 4 --num-workers 1 --prefetch-factor 10 --ntrain 10000 --nvalid 10000
#    --train --conv-type mamba --num-epochs 20 --gpu-batch-multiplier 10 --num-workers 1 --prefetch-factor 10 --ntrain 10000 --nvalid 10000

# singularity exec -B /scratch/persistent --nv \
#     --env PYTHONPATH=hep_tfds \
#     $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
#     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
#     --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --num-workers 1 --prefetch-factor 10 --load experiments/pyg-cms_20240204_183048_293390/sub1/best_weights.pth --ntest 1000
