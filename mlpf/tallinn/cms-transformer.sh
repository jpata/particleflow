#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 4
#SBATCH --mem-per-gpu=8G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.10.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch-persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c parameters/cms-transformer.yaml --plot-freq 1 --num-cpus 8 --batch-multiplier 2 \
    --weights experiments/cms-transformer_20221211_190751_326177.gpu0.local/weights/weights-06-0.954738.hdf5
