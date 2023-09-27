#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:a100:2
#SBATCH --mem-per-gpu=40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.13.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c parameters/cms-gen.yaml --plot-freq 1 --num-cpus 32 --batch-multiplier 2
