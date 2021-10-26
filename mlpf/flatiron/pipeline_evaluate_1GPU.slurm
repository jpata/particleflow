#!/bin/sh

# Walltime limit
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --constraint=a100

# Job name
#SBATCH -J pipeeval

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

module purge
module load slurm gcc cuda/11.1.0_455.23.05 cudnn/v8.0.4-cuda-11.1
nvidia-smi

source ~/miniconda3/bin/activate tf2
which python3
python3 --version

echo 'Starting evaluation.'
CUDA_VISIBLE_DEVICES=0 python3 mlpf/pipeline.py evaluate -c $1 -t $2
echo 'Evaluation done.'
