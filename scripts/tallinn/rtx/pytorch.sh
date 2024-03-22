#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:4
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-03-11

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset delphes --gpus 4 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-delphes.yaml \
    --train --test --make-plots --conv-type attention --gpu-batch-multiplier 1 --num-workers 1 --prefetch-factor 10 --attention-type efficient --dtype float32

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset clic --gpus 4 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
    --train --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --num-workers 1 --prefetch-factor 10 --attention-type math --dtype float32

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset clic_hits --gpus 4 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic-hits.yaml \
    --train --test --make-plots --conv-type attention --gpu-batch-multiplier 1 --num-workers 1 --prefetch-factor 10 --attention-type efficient --dtype float32

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 4 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --test --make-plots --conv-type attention --gpu-batch-multiplier 1 --num-workers 1 --prefetch-factor 10 --attention-type efficient --dtype float32
