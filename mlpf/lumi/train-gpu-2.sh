#!/bin/bash
#SBATCH --job-name=mlpf-train-cms-gen
#SBATCH --account=project_465000301
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:mi250:2
#SBATCH --partition=standard-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/users/patajoos/tf-rocm.simg
cd ~/particleflow

#TF training
singularity exec \
    --rocm \
    -B /scratch/project_465000301 \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/project_465000301/tensorflow_datasets \
    $IMG python3 mlpf/pipeline.py train \
    --config parameters/clic-hits.yaml --plot-freq 1 --num-cpus 16 \
    --batch-multiplier 2 --ntrain 20000 --ntest 20000
