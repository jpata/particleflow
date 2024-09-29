#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 200G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-18
cd ~/particleflow

WEIGHTS=experiments/pyg-clic_20240921_171345_363520/checkpoints/checkpoint-27-1.650469.pth
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG  python3 mlpf/pyg_pipeline.py --dataset clic --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
     --test --make-plots --gpu-batch-multiplier 100 --load $WEIGHTS --dtype bfloat16 --prefetch-factor 10 --num-workers 8 --load experiments/pyg-clic_20240928_174207_681758/checkpoints/checkpoint-27-2.987056.pth --ntest 10000
