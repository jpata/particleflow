#!/bin/sh

#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus 4
#SBATCH --constraint=a100

# Job name
#SBATCH -J find_lr

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

# module --force purge; module load modules/2.1-alpha1
# module load slurm gcc/11.3.0 nccl cuda/11.8.0 cudnn/8.4.0.27-11.6
module --force purge; module load modules/2.0-20220630
module load slurm gcc cmake/3.22.3 nccl cuda/11.4.4 cudnn/8.2.4.15-11.4 openmpi/4.0.7
nvidia-smi

source ~/miniconda3/bin/activate tf2
which python3
python3 --version

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py find-lr -c $1
