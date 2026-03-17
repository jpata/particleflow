#!/bin/bash
export PF_SITE=local
DATA_DIR=$(python3 scripts/get_param.py particleflow_spec.yaml productions.cms_run3.workspace_dir)/tfds/
./scripts/local/wrapper.sh \
    python mlpf/pipeline.py \
    --spec-file particleflow_spec.yaml \
    --model-name pyg-cms-v1 \
    --production cms_run3 \
    --data-dir $DATA_DIR \
    train \
    --gpu_batch_multiplier 8 \
    --checkpoint_freq 100 \
    --val_freq 100 \
    --ntrain 1000 \
    --nvalid 100 \
    --num_steps 200 \
    --model.type attention \
    --num_workers 1
