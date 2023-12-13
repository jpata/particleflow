#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.14.0.simg
cd ~/particleflow

#change these
EXPDIR=experiments/cms-gen_20231206_182649_456797.gpu1.local
WEIGHTS=experiments/cms-gen_20231206_182649_456797.gpu1.local/weights/weights-01-4.213115.hdf5

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python3.10 mlpf/pipeline.py evaluate \
    --train-dir $EXPDIR --weights $WEIGHTS

#singularity exec -B /scratch/persistent --nv \
#    --env PYTHONPATH=hep_tfds \
#    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
#    $IMG python3.10 mlpf/pipeline.py plots \
#    --train-dir $EXPDIR
