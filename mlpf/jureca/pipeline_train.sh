#!/bin/sh

#SBATCH --account=raise-ctp2
#SBATCH --partition=dc-gpu
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4

# Job name
#SBATCH -J pipetrain

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

# module --force purge
# module load Stages/2020
# module load GCC CMake NCCL CUDA cuDNN OpenMPI
# # module load GCC/10.3.0 CUDA/11.0 cuDNN/8.0.2.39-CUDA-11.0

module purge
# module load Stages/2020
module load GCC GCCcore/.11.2.0 CMake NCCL CUDA cuDNN OpenMPI
# module load GCC/10.3.0 CUDA/11.0 cuDNN/8.0.2.39-CUDA-11.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

jutil env activate -p raise-ctp2

nvidia-smi

source /p/project/raise-ctp2/cern/miniconda3/bin/activate tf2
echo "Python used:"
which python3
python3 --version

# mkdir $SCRATCH/particleflow
# rsync -ar --exclude={".git","experiments"} . $SCRATCH/particleflow/
# cd $SCRATCH/particleflow
# if [ $? -eq 0 ]
# then
#   echo "Successfully changed directory"
# else
#   echo "Could not change directory" >&2
#   exit 1
# fi

echo 'Starting training.'
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py train -c $1 -p $2 --comet-offline
echo 'Training done.'

# rsync -a experiments/ $SLURM_SUBMIT_DIR/experiments/
