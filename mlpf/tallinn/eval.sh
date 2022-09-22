#!/bin/bash

singularity exec --nv -B /scratch-persistent --env PYTHONPATH=hep_tfds --env TFDS_DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets /home/software/singularity/tf-2.9.0.simg \
    python3 mlpf/pipeline.py evaluate \
    -t experiments/cms-gen_20220920_235422_747201.gpu0.local \
    -w experiments/cms-gen_20220920_235422_747201.gpu0.local/weights/weights-68-2.259857.hdf5 --nevents 5000
