#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.14.0.simg
cd ~/particleflow

#change these
EXPDIR=experiments/clic-hits-lumi_20240113_100022_943210.nid005026
WEIGHTS=experiments/clic-hits-lumi_20240113_100022_943210.nid005026/weights/weights-01-0.201640.hdf5

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python3.10 mlpf/pipeline.py evaluate \
    --train-dir $EXPDIR --weights $WEIGHTS

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python3.10 mlpf/pipeline.py plots \
    --train-dir $EXPDIR
