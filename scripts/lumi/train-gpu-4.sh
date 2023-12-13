#!/bin/bash
#SBATCH --job-name=mlpf-train-cms
#SBATCH --account=project_465000301
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=130G
#SBATCH --gpus-per-task=4
#SBATCH --partition=small-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

cd /scratch/project_465000301/particleflow

module load LUMI/22.08 partition/G

export IMG=/scratch/project_465000301/tf-rocm5.6-tf2.12.simg
export PYTHONPATH=hep_tfds
export TFDS_DATA_DIR=/scratch/project_465000301/tensorflow_datasets
#export MIOPEN_DISABLE_CACHE=true
export MIOPEN_USER_DB_PATH=/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export TF_CPP_MAX_VLOG_LEVEL=-1 #to suppress ROCm fusion is enabled messages
export ROCM_PATH=/opt/rocm
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
    $IMG python3 mlpf/pipeline.py train \
    --config parameters/cms-gen.yaml --plot-freq 1 --num-cpus 8 \
    --batch-multiplier 4 --plot-freq -1
