#!/bin/bash
unset ROCR_VISIBLE_DEVICES
$WITH_CONDA
source particleflow-env/bin/activate
pip list installed
rocm-smi --showdriverversion

python mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-hits-v1 \
  --production cld \
  --data-dir $TFDS_DATA_DIR \
  train \
  --num_steps 100000 \
  --checkpoint_freq 100 \
  --val_freq 500 \
  --gpus 8 \
  --dtype bfloat16 \
  --num_workers 2 \
  --prefetch_factor 2 \
  --gpu_batch_multiplier 16 \
  --compile \
  --lr 0.001 \
  --optimizer lamb
