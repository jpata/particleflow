#!/bin/sh

# Walltime limit
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1

#SBATCH --mem=1000G
#SBATCH --cpus-per-task=112
#SBATCH -p eval
#SBATCH --constraint=sapphire
#SBATCH -w worker6302

# #SBATCH -p gpu
# #SBATCH --constraint=v100
# #SBATCH --gpus-per-task=1
# #SBATCH -w workergpu094

# Job name
#SBATCH -J pipeinfer

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

train_dir="/mnt/ceph/users/ewulff/particleflow/experiments/bsm10_1GPU_clic-gnn-tuned-v130_20230724_035617_375578.workergpu037"

# export CUDA_VISIBLE_DEVICES=0

## declare an array variable
declare -a bs=(1024 512 256 128 64 32 16 8 4 2 1)

## now loop through the above array
for i in "${bs[@]}"
do
    echo 'Starting inference.'
    python3 mlpf/pipeline.py infer \
        --train-dir $train_dir \
        --nevents 4000 \
        --bs "$i" \
        --num-runs 11 \
        -o /mnt/ceph/users/ewulff/particleflow/inference_tests/results_$SLURMD_NODENAME.json
    echo 'Inference done.'
done
