#!/bin/bash

export PF_SITE=local
DATA_DIR=$(python3 scripts/get_param.py particleflow_spec.yaml productions.cms_run3.workspace_dir)/tfds/
./scripts/local/wrapper.sh python scripts/cms-validate-onnx.py \
    --checkpoint experiments/pyg-cms-v1_cms_run3_20260313_185515_876757/checkpoints/checkpoint-1000.pth \
    --model-kwargs experiments/pyg-cms-v1_cms_run3_20260313_185515_876757/model_kwargs.pkl \
    --dataset cms_pf_ttbar \
    --data-dir $DATA_DIR \
    --num-events 500 \
    --outdir ./onnx_validation_cms --device cuda
