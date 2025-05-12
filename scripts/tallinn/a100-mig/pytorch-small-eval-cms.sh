#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 100G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-12-03
cd ~/particleflow

WEIGHTS=experiments/pyg-cms_20250510_121014_108409/checkpoints/checkpoint-20-3.352647.pth
EXPDIR=experiments/pyg-cms_20250510_121014_108409/
DATASET=cms_pf_ttbar
env
singularity exec -B /local -B /scratch/persistent --nv \
     --env PYTHONPATH=`pwd` \
     --env KERAS_BACKEND=torch \
     $IMG python mlpf/pipeline.py --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --gpu-batch-multiplier 2 --load $WEIGHTS --ntest 10000 --dtype bfloat16 --num-workers 1 --prefetch-factor 10 --test-datasets $DATASET --experiment-dir $EXPDIR
