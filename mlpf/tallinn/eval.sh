#!/bin/bash

EXP=experiments/cms-gen_20220910_185642_456887.gpu0.local/
singularity exec --nv -B /scratch-persistent --env PYTHONPATH=hep_tfds --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets /home/software/singularity/tf-2.9.0.simg \
    python3 mlpf/pipeline.py evaluate -t $EXP --nevents 1000
