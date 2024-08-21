#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 50G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-02
cd ~/particleflow

WEIGHTS=experiments/pyg-cms-ttbar-nopu_20240816_143034_376064/checkpoints/checkpoint-30-563.293516.pth
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG  python mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms-ttbar-nopu.yaml \
     --test --make-plots --gpu-batch-multiplier 10 --load $WEIGHTS --ntest 5000 --dtype bfloat16
