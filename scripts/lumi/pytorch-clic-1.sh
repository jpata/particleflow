#!/bin/bash
#SBATCH --job-name=mlpf-train
#SBATCH --account=project_465000301
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-task=1
#SBATCH --partition=small-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

cd /scratch/project_465000301/particleflow

module load LUMI/24.03 partition/G

export IMG=/scratch/project_465000301/pytorch-rocm6.2.simg
export PYTHONPATH=hep_tfds
export TFDS_DATA_DIR=/scratch/project_465000301/tensorflow_datasets
#export MIOPEN_DISABLE_CACHE=true
export MIOPEN_USER_DB_PATH=/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export TF_CPP_MAX_VLOG_LEVEL=-1 #to suppress ROCm fusion is enabled messages
export ROCM_PATH=/opt/rocm
#export NCCL_DEBUG=INFO
#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=4
export KERAS_BACKEND=torch

env

#TF training
singularity exec \
    --rocm \
    -B /scratch/project_465000301 \
    -B /tmp \
    --env LD_LIBRARY_PATH=/opt/rocm/lib/ \
    --env CUDA_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
     $IMG python3 mlpf/pyg_pipeline.py --dataset clic --gpus 1 \
     --data-dir $TFDS_DATA_DIR --config parameters/pytorch/pyg-clic.yaml \
     --train --gpu-batch-multiplier 256 --num-workers 8 --prefetch-factor 100 --checkpoint-freq 1 --conv-type attention --dtype bfloat16 --lr 0.001
