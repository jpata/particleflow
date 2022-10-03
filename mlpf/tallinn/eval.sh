#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=10G
#SBATCH -o logs/slurm-%x-%j-%N.out

singularity exec --nv -B /scratch-persistent --env PYTHONPATH=hep_tfds --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets /home/software/singularity/tf-2.9.0.simg \
    python3 mlpf/pipeline.py evaluate \
    -t experiments/cms-gen_20220923_163529_426249.gpu0.local \
    -w experiments/cms-gen_20220923_163529_426249.gpu0.local/weights/weights-18-2.223574.hdf5 --nevents 5000
