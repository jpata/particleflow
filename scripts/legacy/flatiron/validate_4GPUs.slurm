#!/bin/sh

# Walltime limit
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus 4
#SBATCH --constraint=a100


# Job name
#SBATCH -J eval

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

module --force purge; module load modules/1.49-20211101
module load slurm gcc nccl cuda/11.3.1 cudnn/8.2.0.53-11.3 openmpi/4.0.6
nvidia-smi

source ~/miniconda3/bin/activate tf2
which python3
python3 --version

mkdir $TMPDIR/particleflow
rsync -ar --exclude=".git" . $TMPDIR/particleflow
cd $TMPDIR/particleflow
if [ $? -eq 0 ]
then
  echo "Successfully changed directory"
else
  echo "Could not change directory" >&2
  exit 1
fi

python3 tf_list_gpus.py

echo 'Starting validation.'
#Run the validation to produce the predictions file
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/launcher.py --action eval --model-spec parameters/delphes-gnn-skipconn.yaml --weights $1
echo 'Valdiation done.'

rsync -a experiments/ /mnt/ceph/users/ewulff/experiments/
