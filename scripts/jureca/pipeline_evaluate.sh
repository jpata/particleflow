#!/bin/sh

#SBATCH --account=raise-ctp2
#SBATCH --partition=dc-gpu
#SBATCH --time 01:59:59
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4

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
# module load Stages/2020
module load GCC GCCcore/.11.2.0 CMake NCCL CUDA cuDNN OpenMPI
# module load GCC/10.3.0 CUDA/11.0 cuDNN/8.0.2.39-CUDA-11.0

export CUDA_VISIBLE_DEVICES=0

jutil env activate -p raise-ctp2
nvidia-smi

source /p/project/raise-ctp2/cern/miniconda3/bin/activate tf2
echo "Python used:"
which python3
python3 --version

echo 'Starting evaluation.'
# CUDA_VISIBLE_DEVICES=0 python3 mlpf/pipeline.py evaluate -t $1
CUDA_VISIBLE_DEVICES=0 python3 mlpf/pipeline.py compute_validation_loss -t $1
echo 'Evaluation done.'

# rsync -a experiments/ $SLURM_SUBMIT_DIR/experiments/
