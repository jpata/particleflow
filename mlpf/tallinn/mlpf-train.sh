#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=8G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.11.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c $1 --plot-freq 1 --num-cpus 16 --batch-multiplier $2 --weights experiments/clic_20230412_155159_717751.gpu1.local/weights/weights-100-9.948204.hdf5
