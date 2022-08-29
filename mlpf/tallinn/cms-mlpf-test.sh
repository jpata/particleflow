#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/tf-2.9.0.simg
cd ~/particleflow

env

#TF training
singularity exec -B /scratch-persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c $1 --plot-freq 1 --ntrain 5000 --ntest 1000
