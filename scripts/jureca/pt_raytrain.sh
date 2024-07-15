#!/bin/sh

#SBATCH --account=jureap57
#SBATCH --partition=dc-gpu-devel
#SBATCH --time 2:00:00
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128

# Job name
#SBATCH -J raytrain

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"


module --force purge
ml Stages/2024 GCC/12.3.0 Python/3.11.3
ml CUDA/12 cuDNN/8.9.5.29-CUDA-12 NCCL/default-CUDA-12 Apptainer-Tools/2024

jutil env activate -p jureap57

source ray_tune_env/bin/activate

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

export CUDA_VISIBLE_DEVICES=0,1,2,3
num_gpus=${SLURM_GPUS_PER_TASK}  # gpus per compute node
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}  # necessary on JURECA for Ray to work

## Limit number of max pending trials
export TUNE_MAX_PENDING_TRIALS_PG=$(($SLURM_NNODES * 4))

## Disable Ray Usage Stats
export RAY_USAGE_STATS_DISABLE=1


################# DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

port=7639

export ip_head="$head_node"i:"$port"
export head_node_ip="$head_node"i

echo "Starting HEAD at $head_node"
# apptainer exec --nv -B /p/project/jureap57/cern \
# apptainer/images/jureca_torch2307.sif \
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus  --block &
sleep 20

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$head_node"i:"$port" --redis-password='5241580000000000' \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
    sleep 10
done
echo All Ray workers started.
# fi
##############################################################################################

echo 'Starting training.'
# when training with Ray Train, --gpus should be equal to toal number of GPUs across the Ray Cluster
# apptainer exec --nv -B /p/project/jureap57/cern/data/tensorflow_datasets,/p/project/jureap57/cern/particleflow \
#  apptainer/images/jureca_torch2307.sif \
python3 -u $PWD/mlpf/pyg_pipeline.py --train --ray-train \
    --config $1 \
    --prefix $2 \
    --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_JOB_NUM_NODES)) \
    --gpus $((SLURM_GPUS_PER_TASK*SLURM_JOB_NUM_NODES)) \
    --gpu-batch-multiplier 8 \
    --num-workers 8 \
    --prefetch-factor 8 \
    --experiments-dir /p/project/jureap57/cern/particleflow/experiments \
    --local \
    --ntrain 50000

echo 'Training done.'
