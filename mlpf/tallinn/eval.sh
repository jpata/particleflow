#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=10G
#SBATCH -o logs/slurm-%x-%j-%N.out

singularity exec --nv -B /scratch-persistent --env PYTHONPATH=hep_tfds --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets /home/software/singularity/tf-2.10.0.simg \
    python3 mlpf/pipeline.py evaluate \
    -t experiments/cms-transformer_20221114_182159_902630.gpu0.local \
    --nevents 5000
