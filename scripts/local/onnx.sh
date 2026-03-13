#!/bin/bash

export PF_SITE=local
DATA_DIR=$(python3 scripts/get_param.py particleflow_spec.yaml productions.cld.workspace_dir)/tfds/
./scripts/local/wrapper.sh python scripts/cms-validate-onnx.py \
    --checkpoint experiments/pyg-cld-v1_cld_20260313_144605_188505/checkpoints/checkpoint-5000.pth \
    --model-kwargs experiments/pyg-cld-v1_cld_20260313_144605_188505/model_kwargs.pkl \
    --dataset cld_edm_ttbar_pf \
    --data-dir $DATA_DIR \
    --num-events 100 \
    --outdir ./onnx_validation_cld --device cuda

