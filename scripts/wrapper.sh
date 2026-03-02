#!/bin/bash
apptainer exec --env PYTHONPATH=`pwd` `./scripts/get_pytorch_image.sh` "$@"
