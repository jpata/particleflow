#!/bin/bash
set -o xtrace
export PF_SITE=tallinn
export KERAS_BACKEND=torch
IMAGE=$(pixi run python3 scripts/get_param.py particleflow_spec.yaml project.container)
BINDS=$(pixi run python3 scripts/get_param.py particleflow_spec.yaml project.bind_mounts)
B_ARGS=""
for b in $BINDS; do
    B_ARGS="$B_ARGS -B $b"
done
apptainer exec --nv $B_ARGS --env PYTHONPATH=`pwd` $IMAGE "$@"
