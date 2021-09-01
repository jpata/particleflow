#!/bin/sh

#SBATCH -t 168:00:00
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --constraint=a100,sxm4
#SBATCH --gpus-per-task=4

# Job name
#SBATCH -J hypertune

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

module purge
module load slurm gcc cuda/11.1.0_455.23.05 cudnn/v8.0.4-cuda-11.1

source ~/miniconda3/bin/activate tf2
which python3
python3 --version


# Getting the node hostnames
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

echo $nodes

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" mlpf/hypertune_scripts/run_chief.sh "chief" $head_node_ip $port $1 "/mnt/ceph/users/ewulff/hypertunes/hypertune_out${2}_${SLURM_JOB_ID}" &> logs_slurm/chief_${SLURM_JOB_ID} &
sleep 5

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    tunerID="tuner$i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        mlpf/hypertune_scripts/run_tuner.sh $tunerID $head_node_ip $port $1 "/mnt/ceph/users/ewulff/hypertunes/hypertune_out${2}_${SLURM_JOB_ID}" &> logs_slurm/tuner_${SLURM_JOB_ID}_${i} &
    sleep 1
done
wait # keep the wait statement, it is important
echo "Done."
