#!/bin/sh

#SBATCH --account=jureap57
#SBATCH --partition=dc-gpu-devel
#SBATCH --time 0:20:00
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=4
#SBATCH --exclusive

# Job name
#SBATCH -J raytune

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

echo "Python used:"
which python3
python3 --version

sleep 1
# make sure CUDA devices are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}  # necessary on JURECA for Ray to work

num_gpus=4

## Limit number of max pending trials
export TUNE_MAX_PENDING_TRIALS_PG=$(($SLURM_NNODES * 4))

## Disable Ray Usage Stats
export RAY_USAGE_STATS_DISABLE=1


################# DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

port=7639

export ip_head="$head_node"i:"$port"
export head_node_ip="$head_node"i

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus  --block &
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$head_node"i:"$port" --redis-password='5241580000000000' \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
    sleep 5
done
echo All Ray workers started.
##############################################################################################

# echo "Starting test..."
# python3 -u $PWD/mlpf/raytune/rayinit.py
# echo "Exited test."

echo 'Starting HPO.'
# when training with Ray Train, --gpus should be equal to toal number of GPUs across the Ray Cluster
python3 -u $PWD/mlpf/pipeline.py --train \
    --data-dir /p/project/jureap57/cern/tensorflow_datasets/clusters \
    --config $1 \
    --hpo $2 \
    --ray-cpus 64 \
    --gpus $num_gpus \
    --num-workers 8 \
    --prefetch-factor 8 \
    --gpu-batch-multiplier 8 \
    --num-epochs 2 \
    --ntrain 5000 \
    --nvalid 5000 \
    --raytune-num-samples 2

echo 'HPO done.'
