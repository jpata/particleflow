#!/bin/bash

SINGULARITY_IMAGE=/storage/user/jpata/gpuservers/singularity/images/over_edge.simg

#singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 test/train_clustering.py
singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 test/train_regression.py
