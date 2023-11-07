#!/bin/sh

# Walltime limit
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --constraint=a100-80gb,ib

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
source ~/miniconda3/bin/activate pytorch
which python3
python3 --version


echo 'Starting training.'
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u mlpf/pyg_pipeline.py \
    --config $1 \
    --load $2 \
    --gpus "0,1,2,3" \
    --conv-type gnn_lsh \
    --gpu-batch-multiplier 4 \
    --num-workers 1 \
    --prefetch-factor 2 \
    --test \
    --make-plots

echo 'Training done.'
