#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:8
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 0,1,2,3,4,5,6,7 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pyg-cms-small.yaml \
    --train --test --make-plots --export-onnx --conv-type gnn_lsh --num-epochs 10 --ntrain 5000 --ntest 5000 --gpu-batch-multiplier 1 --num-workers 1 --prefetch-factor 10
