#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=10G
#SBATCH -o logs/slurm-%x-%j-%N.out

#singularity exec --nv -B /scratch-persistent --env PYTHONPATH=hep_tfds --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets /home/software/singularity/tf-2.9.0.simg \
#    python3 mlpf/pipeline.py evaluate \
#    -t experiments/cms-gen_20220923_163529_426249.gpu0.local \
#    -w experiments/cms-gen_20220923_163529_426249.gpu0.local/weights/weights-18-2.223574.hdf5 --nevents 5000

singularity exec --nv -B /scratch-persistent --env PYTHONPATH=hep_tfds --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets /home/software/singularity/tf-2.9.0.simg \
    python3 mlpf/pipeline.py evaluate \
    -t experiments/epochs60_restarted_at_epoch47_nocomet_cms-gen-best221005_20221010_064016_294510.workergpu040 \
    --nevents 5000
