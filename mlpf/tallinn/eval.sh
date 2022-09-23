#!/bin/bash

singularity exec --nv -B /scratch-persistent --env PYTHONPATH=hep_tfds --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets /home/software/singularity/tf-2.9.0.simg \
    python3 mlpf/pipeline.py evaluate \
    -t experiments/cms-gen_20220914_141532_570769.gpu0.local \
    -w experiments/cms-gen_20220914_141532_570769.gpu0.local/weights/weights-30-2.766837.hdf5 --nevents 5000
