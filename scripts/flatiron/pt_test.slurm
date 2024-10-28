#!/bin/sh

# Walltime limit
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --constraint=a100-80gb&sxm4

# Job name
#SBATCH -J pt_test

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"


module --force purge; module load modules/2.2-20230808
module load slurm gcc cmake cuda/12.1.1 cudnn/8.9.2.26-12.x nccl openmpi apptainer

nvidia-smi
source ~/miniforge3/bin/activate mlpf
which python3
python3 --version


echo 'Starting testing.'
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u mlpf/pipeline.py \
    --config $1 \
    --load $2 \
    --gpus 4 \
    --gpu-batch-multiplier 8 \
    --num-workers 4 \
    --prefetch-factor 8 \
    --test \
    --make-plots

echo 'Testing done.'
