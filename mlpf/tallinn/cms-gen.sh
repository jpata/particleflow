#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 8
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/tf-2.8.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch-persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c parameters/cms-gen.yaml --plot-freq 100 \
    -c experiments/cms-gen_20220503_145445_570900.gpu0.local/config.yaml \
    -w experiments/cms-gen_20220503_145445_570900.gpu0.local/weights/weights-100-2.682420.hdf5
