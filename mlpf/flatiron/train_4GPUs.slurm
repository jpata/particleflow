#!/bin/sh

# Walltime limit
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus 4
#SBATCH --constraint=a100

# Job name
#SBATCH -J train

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

module purge
module load slurm gcc cuda/11.1.0_455.23.05 cudnn/v8.0.4-cuda-11.1
nvidia-smi

source ~/miniconda3/bin/activate tf2
which python3
python3 --version

mkdir $TMPDIR/particleflow
rsync -ar --exclude={".git","experiments"} . $TMPDIR/particleflow
cd $TMPDIR/particleflow
if [ $? -eq 0 ]
then
  echo "Successfully changed directory"
else
  echo "Could not change directory" >&2
  exit 1
fi

python3 tf_list_gpus.py

echo 'Starting training.'
# Run the training of the base GNN model using e.g. 4 GPUs in a data-parallel mode
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/launcher.py --action train --model-spec $1
echo 'Training done.'
ls -l experiments

rsync -a experiments/ /mnt/ceph/users/ewulff/experiments/
