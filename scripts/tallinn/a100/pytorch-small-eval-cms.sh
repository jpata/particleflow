#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
<<<<<<< HEAD
#SBATCH --mem-per-gpu 150G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-07-08
cd ~/particleflow

WEIGHTS=experiments/pyg-cms_20240804_095032_809397/checkpoints/checkpoint-13-19.827532.pth
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG  python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --gpu-batch-multiplier 20 --load $WEIGHTS --ntest 50000 --dtype bfloat16
=======
#SBATCH --mem-per-gpu 50G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-18
cd ~/particleflow

WEIGHTS=experiments/pyg-cms_20240915_162455_135826/checkpoints/checkpoint-01-4.121693.pth
env
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG python mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --gpu-batch-multiplier 1 --load $WEIGHTS --ntrain 1000 --ntest 1000 --nvalid 1000 --dtype bfloat16 --num-epochs 20
>>>>>>> origin/fixes_sep6
