#!/bin/sh

# Walltime limit
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --constraint=a100-80gb,ib

# Job name
#SBATCH -J pt_train

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
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pyg_pipeline.py --train \
    --config $1 \
    --prefix $2 \
    --gpus "0" \
    --num_epochs 3 \
    --lr 0.0001 \
    --conv_type gnn_lsh \
    --ntrain 400 \
    --ntest 400

echo 'Training done.'
