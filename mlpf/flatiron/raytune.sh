#!/bin/bash

#SBATCH -t 168:00:00
#SBATCH -N 12
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --constraint=a100,sxm4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64

# Job name
#SBATCH -J raytune

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

export TUNE_RESULT_DIR="/mnt/ceph/users/ewulff/ray_results/tune_result_dir"
export TUNE_MAX_PENDING_TRIALS_PG=${SLURM_NNODES}

module purge
module load slurm gcc cuda/11.1.0_455.23.05 cudnn/v8.0.4-cuda-11.1
nvidia-smi

source ~/miniconda3/bin/activate tf2
which python3
python3 --version


################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password
echo "Redis password: ${redis_password}"

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]} 
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 mlpf/raytune/start-head.sh $ip $SLURM_JOB_ID $2 "/mnt/ceph/users/ewulff/nvidia_smi_logs/" &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i mlpf/raytune/start-worker.sh $ip_head $SLURM_JOB_ID $i $2 "/mnt/ceph/users/ewulff/nvidia_smi_logs/" &
  sleep 5
done
##############################################################################################

#### call your code below
python3 mlpf/pipeline.py raytune -c $1 -n $2 --cpus "${SLURM_CPUS_PER_TASK}" --gpus "${SLURM_GPUS_PER_TASK}"
exit
