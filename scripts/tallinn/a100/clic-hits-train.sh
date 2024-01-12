#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 100G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.14.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python3.10 mlpf/pipeline.py train -c parameters/clic-hits.yaml \
    --plot-freq 1 --num-cpus 32 --batch-multiplier 2 --weights experiments/clic-hits_20240106_194456_549323.gpu1.local/weights/weights-01-1.005839.hdf5
