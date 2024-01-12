#!/bin/bash
#SBATCH --job-name=mlpf-train-cms
#SBATCH --account=project_465000301
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=130G
#SBATCH --gpus-per-task=4
#SBATCH --partition=small-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

cd /scratch/project_465000301/particleflow

module load LUMI/22.08 partition/G

export IMG=/scratch/project_465000301/pytorch-rocm.simg
export PYTHONPATH=hep_tfds
export TFDS_DATA_DIR=/scratch/project_465000301/tensorflow_datasets
#export MIOPEN_DISABLE_CACHE=true
export MIOPEN_USER_DB_PATH=/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export ROCM_PATH=/opt/rocm
#export NCCL_DEBUG=WARN
#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=4

env

singularity exec --rocm \
  -B /scratch/project_465000301 \
  -B /tmp \
  --env PYTHONPATH=hep_tfds \
  $IMG python3 mlpf/pyg_pipeline.py --dataset cms --gpus $SLURM_GPUS_PER_TASK \
  --data-dir $TFDS_DATA_DIR --config parameters/pyg-cms.yaml \
  --train \
  --conv-type gnn_lsh \
  --num-epochs 20 --gpu-batch-multiplier 4 --num-workers 1 --prefetch-factor 5 --checkpoint-freq 1
