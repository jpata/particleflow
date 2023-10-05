#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu=40G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/tf-2.14.0.simg
cd ~/particleflow

#TF training
singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env TFDS_DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets \
    $IMG python3.10 mlpf/pipeline.py train -c parameters/cms-gen.yaml --plot-freq 1 --num-cpus 32 --batch-multiplier 2 \
    --weights experiments/cms-gen_20230926_205923_762855.gpu1.local/weights/weights-32-0.418947.hdf5
