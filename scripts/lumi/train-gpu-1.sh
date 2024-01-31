#!/bin/bash
#SBATCH --job-name=mlpf-train-cms
#SBATCH --account=project_465000301
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gpus-per-task=1
#SBATCH --partition=small-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

cd /scratch/project_465000301/particleflow

module load LUMI/22.08 partition/G

export IMG=/scratch/project_465000301/tf-rocm5.6-tf2.12-2024-01-11.simg
export PYTHONPATH=hep_tfds
export TFDS_DATA_DIR=/scratch/project_465000301/tensorflow_datasets
#export MIOPEN_DISABLE_CACHE=true
export MIOPEN_USER_DB_PATH=/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export TF_CPP_MAX_VLOG_LEVEL=-1 #to suppress ROCm fusion is enabled messages
export ROCM_PATH=/opt/rocm
export NCCL_DEBUG=WARN
#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=4

singularity exec \
    --env LD_LIBRARY_PATH=/opt/rocm/lib/ \
    --rocm $IMG rocm-smi --showdriverversion --showmeminfo vram

#TF training
singularity exec \
    --rocm \
    -B /scratch/project_465000301 \
    -B /tmp \
    --env LD_LIBRARY_PATH=/opt/rocm/lib/ \
    $IMG python3.9 mlpf/pipeline.py hypertune --config parameters/bench/clic-hits-bench.yaml --ntrain 100 --ntest 100 -o hypertuning_100
