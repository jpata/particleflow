#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:4
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2023-12-06

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 4 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pyg-cms-small.yaml \
    --train --test --make-plots --conv-type gnn_lsh --num-epochs 20 --gpu-batch-multiplier 1 --num-workers 1 --prefetch-factor 10

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 4 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pyg-cms-small.yaml \
    --train --test --make-plots --conv-type gravnet --num-epochs 20 --gpu-batch-multiplier 1 --num-workers 1 --prefetch-factor 10

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 4 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pyg-cms.yaml \
    --train --test --make-plots --conv-type mamba --num-epochs 50 --gpu-batch-multiplier 5 --num-workers 1 --prefetch-factor 10

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 4 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pyg-cms.yaml \
    --train --test --make-plots --conv-type attention --num-epochs 20 --gpu-batch-multiplier 5 --num-workers 1 --prefetch-factor 10


#Eval
#singularity exec -B /scratch/persistent --nv \
#    --env PYTHONPATH=hep_tfds \
#    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 4 \
#    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pyg-cms.yaml \
#    --test --make-plots --conv-type mamba --gpu-batch-multiplier 5 --num-workers 1 --prefetch-factor 10 --load experiments/pyg-cms_20231211_115702_368383/sub1/best_weights.pth --ntest 1000
