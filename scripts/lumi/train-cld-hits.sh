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
  --gpus 8 \
  --dtype bfloat16 \
  --num_workers 2 \
  --prefetch_factor 2 \
  --pad_to_multiple_elements 100 \
  --gpu_batch_multiplier 32 \
  --model.attention.use_jagged_attention True \
  --model.attention.use_flash_attn_varlen True \
  --compile \
  --model.attention.num_convs 6 \
  --model.type attention \
  --model.task_queries false \
  --lr 0.001 --num_steps 10000 --val_freq 1000 --checkpoint_freq 1000

#  --compile \
