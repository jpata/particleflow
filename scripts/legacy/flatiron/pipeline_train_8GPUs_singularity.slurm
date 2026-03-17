#!/bin/sh

# Walltime limit
#SBATCH -t 0-04:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-task=8
#SBATCH --constraint=h100,ib
# #SBATCH --mem 256G

# Job name
#SBATCH -J pipetrain

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH
module load cuda/12 cudnn/cuda12 nccl/cuda12 singularity  # these names are specific to gkrawezik's modules
nvidia-smi

# ensure CPU is keeping private threads for scheduling operations on the GPUs
# https://www.tensorflow.org/guide/gpu_performance_analysis#2_gpu_host_thread_contention
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nvidia-smi

echo 'Starting training.'
singularity run --nv -B /mnt/ceph/users/ewulff/tensorflow_datasets,/mnt/ceph/users/ewulff/particleflow \
 tensorflow_23.05-tf2-py3.sif \
 python3 $PWD/mlpf/pipeline.py train -c $1 -p $2 \
 --seeds --comet-exp-name particleflow-tf-clic --benchmark_dir exp_dir
echo 'Training done.'
