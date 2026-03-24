#!/bin/sh

# Walltime limit
#SBATCH -t 2:00:00
#SBATCH -N 2
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --constraint=h100

# Job name
#SBATCH -J rayhpo

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
RAY_LOCAL_FLAG=""

if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
  ################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
  redis_password=$(uuidgen)
  export redis_password
  echo "Redis password: ${redis_password}"

  nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
  nodes_array=( $nodes )

  node_1=${nodes_array[0]}
  ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address
  port=6379
  ip_head=$ip:$port
  export ip_head
  export head_node_ip=$ip  # needed by _init_ray in distributed_ray.py
  echo "ip_head: $ip_head"
  echo "head_node_ip: $head_node_ip"

  echo "STARTING HEAD at $node_1"
  srun --nodes=1 --ntasks=1 -w $node_1 \
    ray start --head --node-ip-address="$node_1" --port=$port \
    --num-cpus $((SLURM_CPUS_PER_TASK)) --num-gpus $num_gpus --block &

  sleep 10

  worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
  for ((  i=1; i<=$worker_num; i++ ))
  do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w $node_i \
      ray start --address "$node_1":"$port" \
      --num-cpus $((SLURM_CPUS_PER_TASK)) --num-gpus $num_gpus --block &
    sleep 5
  done

  echo "All Ray workers started."
  ##############################################################################################
else
  # Single node: use --ray-local so pipeline.py calls ray.init() locally
  RAY_LOCAL_FLAG="--ray-local"
fi


DATA_DIR="/mnt/ceph/users/ewulff/tensorflow_datasets/cms"

# HPO experiment name (passed as first argument, or use a default)
HPO_NAME=${1:-"hpo_cms_h100"}

# Number of GPUs per trial (each trial trains on this many GPUs)
GPUS_PER_TRIAL=${2:-$num_gpus}

# Set number of CPUs such that all cpus are used when running max number of parallel trials given the number of GPUs per trial. This is important to ensure good performance of Ray Tune.
CPUS_PER_TRIAL=$((SLURM_CPUS_PER_TASK * GPUS_PER_TRIAL / num_gpus))

# Number of HPO samples/trials to run
NUM_SAMPLES=${3:-5}


echo "Starting HPO: name=$HPO_NAME, gpus_per_trial=$GPUS_PER_TRIAL, cpus_per_trial=$CPUS_PER_TRIAL, num_samples=$NUM_SAMPLES"
python3 -u mlpf/pipeline.py \
    --spec-file particleflow_spec.yaml --model-name pyg-cms-v1 --production cms_run3 \
    --data-dir $DATA_DIR \
    ray-hpo \
    --name $HPO_NAME \
    --ray-gpus $GPUS_PER_TRIAL \
    --ray-cpus $CPUS_PER_TRIAL \
    --raytune-num-samples $NUM_SAMPLES \
    $RAY_LOCAL_FLAG \
    --raytune.asha.grace_period 2 \
    --checkpoint_freq 100 \
    --val_freq 100

echo 'HPO done.'
