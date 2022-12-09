#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=10G
#SBATCH -o logs/slurm-%x-%j-%N.out

singularity exec --nv -B /scratch-persistent --env PYTHONPATH=hep_tfds --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets /home/software/singularity/tf-2.10.0.simg \
    python3 mlpf/pipeline.py evaluate \
    --train-dir experiments/cms-gen_20220923_163529_426249.gpu0.local
