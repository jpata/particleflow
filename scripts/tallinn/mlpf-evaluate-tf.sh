#!/bin/bash

IMG=/home/software/singularity/tf-2.11.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch-persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py evaluate --train-dir $1
