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
    $IMG python mlpf/pipeline.py train -c parameters/cms-gen.yaml --plot-freq 1 --num-cpus 8 --batch-multiplier 2 \
    --weights experiments/cms-gen_20221211_231119_254381.gpu0.local/weights/weights-02-0.955068.hdf5
