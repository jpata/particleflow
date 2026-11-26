#!/bin/bash
unset ROCR_VISIBLE_DEVICES
$WITH_CONDA
source particleflow-env/bin/activate
pip list installed
rocm-smi --showdriverversion

python mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-hits-heptv2-set-v1 \
  --production cld \
  --data-dir $TFDS_DATA_DIR \
  train \
  --gpus 8 \
  --dtype bfloat16 \
  --num_workers 2 \
  --prefetch_factor 2 \
  --gpu_batch_multiplier 8 \
  --model.heptv2.activation_checkpointing true \
  --lr 0.001 --num_steps 10000 --val_freq 1000 --checkpoint_freq 1000

#  --compile \
