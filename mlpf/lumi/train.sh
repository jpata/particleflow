#!/bin/bash
#SBATCH --job-name=mlpf-train-cms-gen
#SBATCH --account=project_465000301
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=standard
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/users/patajoos/tf-2.10.0.simg
cd ~/particleflow
env

#TF training
singularity exec -B /scratch/project_465000301 \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/project_465000301/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c parameters/cms-gen.yaml --plot-freq 1 --num_cpus 16 --nepochs 5
