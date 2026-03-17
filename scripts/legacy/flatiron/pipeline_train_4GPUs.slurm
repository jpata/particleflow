#!/bin/sh

# Walltime limit
#SBATCH -t 7-0:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --constraint=a100-80gb,ib

# Job name
#SBATCH -J pipetrain

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

# module --force purge; module load modules/2.0-20220630
# module load slurm gcc cmake/3.22.3 nccl cuda/11.4.4 cudnn/8.2.4.15-11.4 openmpi/4.0.7

module --force purge; module load modules/2.1.1-20230405
module load slurm gcc cmake nccl cuda/11.8.0 cudnn/8.4.0.27-11.6 openmpi/4.0.7

nvidia-smi

source ~/miniconda3/bin/activate tf2
which python3
python3 --version

# make tensorflow find cupti (needed for profiling)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/sw/nix/store/3xpm36w2kcri3j1m5j15hg025my1p4kx-cuda-11.8.0/extras/CUPTI/lib64/

export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

echo 'Starting training.'
# Run the training of the base GNN model using e.g. 4 GPUs in a data-parallel mode
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py train -c $1 -p $2 \
    --seeds --comet-exp-name particleflow-tf-clic
echo 'Training done.'
