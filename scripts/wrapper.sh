#!/bin/bash
apptainer exec --nv --env PYTHONPATH=`pwd` `./scripts/get_pytorch_image.sh` "$@"
