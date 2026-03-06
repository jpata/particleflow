#!/bin/bash
apptainer exec --nv -B /afs -B /eos -B /run --env KRB5CCNAME=$KRB5CCNAME --env PYTHONPATH=`pwd` `./scripts/get_pytorch_image.sh` "$@"
