#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.11.0.simg
cd ~/particleflow

EXPDIR=experiments/clic-hits_20230512_161010_875811.gpu1.local
WEIGHTS=experiments/clic-hits_20230512_161010_875811.gpu1.local/weights/weights-06-0.076698.hdf5
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py evaluate \
    --train-dir $EXPDIR --weights $WEIGHTS

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py plots \
    --train-dir $EXPDIR
