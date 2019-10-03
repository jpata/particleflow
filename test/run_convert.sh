#!/bin/bash

SINGULARITY_IMAGE=/storage/group/gpu/software/singularity/ibanks/edge.simg

singularity exec --nv -B /storage $SINGULARITY_IMAGE python3 test/convert.py
