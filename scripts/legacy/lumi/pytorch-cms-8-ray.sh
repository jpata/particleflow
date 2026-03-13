#!/bin/bash
#SBATCH --job-name=mlpf-train
#SBATCH --account=project_465001293
#SBATCH --time=1-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gpus-per-task=8
#SBATCH --partition=standard-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

set +x
set +e

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
module load aws-ofi-rccl

cd /scratch/project_465001293/joosep/particleflow

export IMG=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0.sif
export PYTHONPATH=`pwd`
export TFDS_DATA_DIR=/scratch/project_465001293/joosep/tensorflow_datasets
#export MIOPEN_DISABLE_CACHE=true
export MIOPEN_USER_DB_PATH=/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export TF_CPP_MAX_VLOG_LEVEL=-1 #to suppress ROCm fusion is enabled messages
#export ROCM_PATH=/opt/rocm
#export NCCL_DEBUG=INFO
#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=4
export KERAS_BACKEND=torch
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=INFO

export redis_password=$(uuidgen)
echo "Redis password: ${redis_password}"

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

export node_1=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address)
# ip=$(srun --nodes=1 --ntasks=1 -w $node_1 host ${node_1}i | awk '{ print $4 }') # making redis-address
sleep 1

export port=6379
export ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head
echo "IP Head: $ip_head"

export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

export num_gpus=16
export num_gpus_task=8

env

echo "Starting HEAD at $ip_head"
srun --nodes=1 --ntasks=1 -w "$node_1" singularity exec -B /scratch/project_465001293 -B /tmp $IMG ./scripts/lumi/ray-head.sh &
sleep 20

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" singularity exec -B /scratch/project_465001293 -B /tmp $IMG ./scripts/lumi/ray-worker.sh &
    sleep 10
done
echo All Ray workers started.

singularity exec \
    -B /scratch/project_465001293 \
    -B /tmp \
    $IMG ./scripts/lumi/ray-train.sh
