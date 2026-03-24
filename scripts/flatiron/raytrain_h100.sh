#!/bin/sh

# Walltime limit
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --constraint=h100

# Job name
#SBATCH -J raytrain

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"


module --force purge; module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer

nvidia-smi
export PYTHONPATH=`pwd`
source ~/miniforge3/bin/activate mlpf
which python3
python3 --version

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=$((SLURM_GPUS_PER_NODE))  # gpus per compute node

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}  # necessary on JURECA for Ray to work

## Disable Ray Usage Stats
export RAY_USAGE_STATS_DISABLE=1

echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_NODEID: $SLURM_NODEID"
echo "DEBUG: SLURM_LOCALID: $SLURM_LOCALID"
echo "DEBUG: SLURM_PROCID: $SLURM_PROCID"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "DEBUG: SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "DEBUG: SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "DEBUG: SLURM_GPUS_PER_TASK: $SLURM_GPUS_PER_TASK"
echo "DEBUG: SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "DEBUG: SLURM_GPUS: $SLURM_GPUS"
echo "DEBUG: num_gpus: $num_gpus"

export RAY_TRAIN_V2_ENABLED=1

DATA_DIR="/mnt/ceph/users/ewulff/tensorflow_datasets/cms"

echo 'Starting training.'
# when training with Ray Train, --gpus should be equal to toal number of GPUs across the Ray Cluster
python3 -u mlpf/pipeline.py \
    --spec-file particleflow_spec.yaml --model-name pyg-cms-v1 --production cms_run3 \
    --data-dir $DATA_DIR \
    --experiments-dir /mnt/home/ewulff/repositories/particleflow/experiments \
    --prefix $1 \
    --comet \
    ray-train \
    --gpus 8 \
    --ray-gpus 8 \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_JOB_NUM_NODES)) \
    --ray-local \
    --gpu_batch_multiplier 32 \
    --comet_name attention_fix

echo 'Training done.'
