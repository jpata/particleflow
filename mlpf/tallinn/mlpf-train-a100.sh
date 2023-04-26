#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.11.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python mlpf/pipeline.py train -c $1 --plot-freq 1 --num-cpus 16 --batch-multiplier $2 --weights experiments/clic-hits_20230413_214732_643093.gpu1.local/weights/weights-10-0.134620.hdf5
