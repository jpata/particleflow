#!/bin/bash
export PF_SITE=lxplus
DATA_DIR=$(python3 scripts/get_param.py particleflow_spec.yaml productions.cld.workspace_dir)/tfds/
./scripts/lxplus/wrapper.sh python3 mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-cld-v1 --production cld --data-dir $DATA_DIR train
