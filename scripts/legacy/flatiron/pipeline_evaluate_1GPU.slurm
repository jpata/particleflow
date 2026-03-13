#!/bin/sh

# Walltime limit
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a100-80gb,ib

# Job name
#SBATCH -J pipeeval

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

module --force purge; module load modules/2.1.1-20230405
module load slurm gcc cmake nccl cuda/11.8.0 cudnn/8.4.0.27-11.6 openmpi/4.0.7

nvidia-smi

source ~/miniconda3/bin/activate tf2
which python3
python3 --version

# make tensorflow find cupti (needed for profiling)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/sw/nix/store/3xpm36w2kcri3j1m5j15hg025my1p4kx-cuda-11.8.0/extras/CUPTI/lib64/

train_dir="experiments/hits_bs16_clic-hits_20230508_064411_129925_RESUMED2_clic-hits_20230522_170633_350485.workergpu064"

echo 'Starting evaluation.'
CUDA_VISIBLE_DEVICES=0 python3 mlpf/pipeline.py evaluate \
    --train-dir $train_dir
echo 'Evaluation done.'

echo 'Starting plotting.'
CUDA_VISIBLE_DEVICES=0 python3 mlpf/pipeline.py plots \
    --train-dir $train_dir
echo 'Plotting done.'
