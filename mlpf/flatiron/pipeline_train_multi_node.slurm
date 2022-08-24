#!/bin/sh

# Walltime limit
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gpus=a100-40gb:4
#SBATCH --constraint=a100,ib
#SBATCH -c 4
#SBATCH --exclusive
# Job name
#SBATCH -J testing

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

module --force purge; module load modules/2.0
module load slurm gcc cmake cudnn cuda openmpi/cuda
#module --force purge; module load modules/1.49-20211101
#module load slurm gcc cmake nccl/2.9.9-1 cuda/11.3.1 cudnn/8.2.0.53-11.3 openmpi/4.0.6

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3

source ~/miniconda/bin/activate tf
which python3
python3 --version

mkdir $TMPDIR/particleflow
rsync -ar --exclude={".git","experiments"} . $TMPDIR/particleflow
cd $TMPDIR/particleflow
if [ $? -eq 0 ]
then
  echo "Successfully changed directory"
else
  echo "Could not change directory" >&2
  exit 1
fi
mkdir experiments

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
node_server=""
for str in ${nodes_array[@]}; do
  node_server=$node_server$str:4,
done
nod=${node_server: : -1}


echo 'Starting training.'
# Run the training of the base GNN model using e.g. 4 GPUs in a data-parallel mode

#horovodrun -np $SLURM_NTASKS -H $nod --log-level DEBUG --tcp --mpi-args="--mca orte_base_help_aggregate 0" python3 mlpf/pipeline.py train -c $1 -p $2

#valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -s mpirun -v -np $SLURM_NTASKS -H $nod -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python3 mlpf/pipeline.py train -c $1 -p $2

#mpirun -v -np $SLURM_NTASKS -H $nod -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python3 mlpf/pipeline.py train -c $1 -p $2

#srun --cpu-bind=none python3 mlpf/pipeline.py train -c $1 -p $2

python3 mlpf/pipeline.py train -c $1 -p $2

echo 'Training done.'
rsync -a experiments/ /mnt/ceph/users/larssorl/experiments/
