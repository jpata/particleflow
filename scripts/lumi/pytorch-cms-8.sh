#!/bin/bash
#SBATCH --job-name=mlpf-train
#SBATCH --account=project_465001293
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=450G
#SBATCH --gpus-per-task=8
#SBATCH --partition=standard-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
module load aws-ofi-rccl

cd /scratch/project_465001293/joosep/particleflow

export IMG=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0.sif
export TFDS_DATA_DIR=/scratch/project_465001293/joosep/tensorflow_datasets
export MIOPEN_USER_DB_PATH=/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export ROCM_PATH=/opt/rocm
export KERAS_BACKEND=torch
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=INFO
export PYTHONPATH=`pwd`
env

singularity exec \
    -B /scratch/project_465001293 \
    -B /tmp \
     $IMG $1
