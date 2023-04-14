#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.11.0.simg
cd ~/particleflow

EXPDIR=experiments/clic_20230412_155159_717751.gpu1.local
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py evaluate \
    --train-dir $EXPDIR

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py plots \
    --train-dir $EXPDIR
