#!/bin/bash
export PF_SITE=lxplus
IMAGE=$(python3 scripts/get_param.py particleflow_spec.yaml project.container)
BINDS=$(python3 scripts/get_param.py particleflow_spec.yaml project.bind_mounts)
B_ARGS=""
for b in $BINDS; do
    B_ARGS="$B_ARGS -B $b"
done
apptainer exec --nv $B_ARGS --env KRB5CCNAME=$KRB5CCNAME --env PYTHONPATH=`pwd` $IMAGE "$@"
