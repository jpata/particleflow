#!/bin/bash

export PF_SITE=local
DATA_DIR=$(python3 scripts/get_param.py particleflow_spec.yaml productions.cms_run3.workspace_dir)/tfds/
uv run --project envs/ort-gpu python3 scripts/cms-validate-onnx.py \
    --checkpoint experiments/pyg-cms-v1_cms_run3_20260313_211827_492562/checkpoints/checkpoint-1000.pth \
    --model-kwargs experiments/pyg-cms-v1_cms_run3_20260313_211827_492562/model_kwargs.pkl \
    --dataset cms_pf_ttbar \
    --data-dir "$DATA_DIR" \
    --num-events 500 --batch-size "${BATCH_SIZE:-4}" --save-inputs --profile \
    --outdir ./onnx_validation_cms --device cuda
