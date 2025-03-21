#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 100G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-12-03
cd ~/particleflow

WEIGHTS=experiments/largebatch_clic_wd3eneg2_gpus4_lr4eneg4_epochs10_pyg-clic-v230_adamw_tunedweightdecay_20250314_085408_738888/checkpoints/checkpoint-10-1.902415.pth
_EXPDIR=experiments/largebatch_clic_wd3eneg2_gpus4_lr4eneg4_epochs10_pyg-clic-v230_adamw_tunedweightdecay_20250314_085408_738888/
singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=`pwd` \
     --env KERAS_BACKEND=torch \
     $IMG  python3 mlpf/pipeline.py --gpus 0 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-clic.yaml \
     --make-plots --ntest 5000 --gpu-batch-multiplier 100 --load $WEIGHTS --dtype bfloat16 --num-workers 0 --experiment-dir $_EXPDIR
