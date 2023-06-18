#!/bin/bash
#SBATCH --job-name=mlpf-train-cms-gen
#SBATCH --account=project_465000301
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --partition=standard-g
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/users/patajoos/tf-rocm.simg
cd ~/particleflow

env

#TF training
singularity exec \
    -B /scratch/project_465000301 \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/project_465000301/tensorflow_datasets \
    $IMG python3 mlpf/pipeline.py train \
    --config parameters/clic-hits.yaml --plot-freq 1 --num-cpus 16 \
    --batch-multiplier 4 --ntrain 20000 --ntest 20000
    
#    --env MIOPEN_USER_DB_PATH=/users/patajoos/miopen-cache \
#    --env MIOPEN_CUSTOM_CACHE_DIR=/users/patajoos/miopen-cache \
#    --env MIOPEN_LOG_LEVEL=7 \
