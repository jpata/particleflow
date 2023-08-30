#!/bin/bash
#SBATCH --job-name=mlpf-train-clic-hits-ln-full
#SBATCH --account=project_465000301
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=130G
#SBATCH --gpus-per-task=7
#SBATCH --partition=small-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

cd /scratch/project_465000301/particleflow

module load LUMI/22.08 partition/G

export IMG=/scratch/project_465000301/tf-rocm.simg
export PYTHONPATH=hep_tfds
export TFDS_DATA_DIR=/scratch/project_465000301/tensorflow_datasets
#export MIOPEN_DISABLE_CACHE=true
export MIOPEN_USER_DB_PATH=/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export TF_CPP_MAX_VLOG_LEVEL=-1 #to suppress ROCm fusion is enabled messages
#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=4

#TF training
singularity exec \
    --rocm \
    -B /scratch/project_465000301 \
    -B /tmp \
    --env LD_LIBRARY_PATH=/opt/rocm/lib/ \
    $IMG python3 mlpf/pipeline.py train \
    --config parameters/clic-test.yaml --plot-freq 1 --num-cpus 8 \
    --batch-multiplier 5 --ntrain 50000 --ntest 50000 --benchmark_dir exp_dir

#    --env MIOPEN_USER_DB_PATH=$MIPEN_USER_DB_PATH \
#    --env MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR \
#    --env MIOPEN_ENABLE_LOGGING=1 \
#    --env MIOPEN_ENABLE_LOGGING_CMD=1 \
#    --env MIOPEN_LOG_LEVEL=7 \
#    --env MIOPEN_ENABLE_LOGGING=1 \
#    --env MIOPEN_ENABLE_LOGGING_CMD=1 \
#    --env MIOPEN_LOG_LEVEL=5 \
