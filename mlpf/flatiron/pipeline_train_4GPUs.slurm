#!/bin/sh

# Walltime limit
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --constraint=a100-80gb

# Job name
#SBATCH -J pipetrain

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

module --force purge; module load modules/2.1-alpha1
module load slurm gcc/11.3.0 nccl cuda/11.8.0 cudnn/8.4.0.27-11.6
nvidia-smi

source ~/miniconda3/bin/activate tf2
which python3
python3 --version


echo 'Starting training.'
# Run the training of the base GNN model using e.g. 4 GPUs in a data-parallel mode
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py train -c $1 -p $2 -s --nepochs 5 -b exp_dir -B 56 --comet-offline
echo 'Training done.'
