#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
<<<<<<< HEAD
#SBATCH --mem-per-gpu 60G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-02
cd ~/particleflow

WEIGHTS=experiments/pyg-clic_20240807_134034_168101/checkpoints/checkpoint-47-9.910686.pth
=======
#SBATCH --mem-per-gpu 50G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-08-18
cd ~/particleflow

WEIGHTS=experiments/pyg-clic_20240910_092302_797928/best_weights.pth
>>>>>>> origin/fixes_sep6
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG  python3 mlpf/pyg_pipeline.py --dataset clic --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
<<<<<<< HEAD
     --test --make-plots --gpu-batch-multiplier 100 --load $WEIGHTS --dtype bfloat16
=======
     --test --make-plots --gpu-batch-multiplier 100 --load $WEIGHTS --dtype bfloat16 --ntest 10000
>>>>>>> origin/fixes_sep6
