#!/bin/bash
#SBATCH --job-name=mlpf-train
#SBATCH --account=project_465001293
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gpus-per-task=8
#SBATCH --partition=standard-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

cd /scratch/project_465001293/joosep/particleflow

#module load LUMI/24.03 partition/G

export IMG=/scratch/project_465001293/joosep/pytorch2.7.1-rocm6.4.1-particleflow.simg
export PYTHONPATH=`pwd`
export TFDS_DATA_DIR=/scratch/project_465001293/joosep/tensorflow_datasets
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

#    --env LD_LIBRARY_PATH=/opt/rocm/lib/ \
#TF training
singularity exec \
    --rocm \
    -B /scratch/project_465001293 \
    -B /tmp \
    --env CUDA_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
     $IMG python3 mlpf/pipeline.py --gpus 8 \
     --data-dir $TFDS_DATA_DIR --config parameters/pytorch/pyg-cms.yaml \
     --train --gpu-batch-multiplier 12 --num-workers 4 --prefetch-factor 10 --checkpoint-freq 1 --conv-type attention --dtype bfloat16 --optimizer lamb --lr 0.002
