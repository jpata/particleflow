#!/bin/sh

#SBATCH --account=raise-ctp2
#SBATCH --partition=dc-gpu
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4

# Job name
#SBATCH -J pipehorovod

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"


module --force purge
module load Stages/2022
module load GCC GCCcore/.11.2.0 CMake NCCL CUDA cuDNN OpenMPI

export CUDA_VISIBLE_DEVICES=0,1,2,3
jutil env activate -p raise-ctp2

sleep 1
nvidia-smi

source /p/project/raise-ctp2/cern/miniconda3/bin/activate tf2
echo "Python used:"
which python3
python3 --version


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


export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
echo $OMP_NUM_THREADS


echo 'Starting training.'
srun --cpu-bind=none python mlpf/pipeline.py train -c $1 -p $2 --comet-offline -j $SLURM_JOBID -m
echo 'Training done.'
