#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 80G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-02-05
cd ~/particleflow

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --conv-type attention --num-epochs 50 --gpu-batch-multiplier 80 --num-workers 2 --prefetch-factor 20 --checkpoint-freq 1 --ntrain 1000 --nvalid 1000 --ntest 1000

# singularity exec -B /scratch/persistent --nv \
#     --env PYTHONPATH=hep_tfds \
#     $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
#     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
#     --test --make-plots --conv-type attention --gpu-batch-multiplier 40 --num-workers 2 --prefetch-factor 20 --load experiments/pyg-cms_20240210_013116_879112/sub1/best_weights.pth --ntest 1000
