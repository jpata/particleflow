#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 300G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-12-03
cd ~/particleflow

ulimit -n 100000
singularity exec -B /scratch/persistent -B /local --nv \
    --env PYTHONPATH=`pwd` \
    --env KERAS_BACKEND=torch \
    $IMG python3 mlpf/pipeline.py --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --conv-type attention --load experiments/pyg-cms_20250517_232752_544969/checkpoints/checkpoint-08-3.863894.pth \
    --gpu-batch-multiplier 5 --checkpoint-freq 1 --num-workers 8 --prefetch-factor 50 --comet --ntest 1000 --test-datasets cms_pf_qcd
