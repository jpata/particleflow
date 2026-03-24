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
  --checkpoint_freq 10000 \
  --val_freq 10000 \
  --gpus 8 \
  --make_plots \
  --dtype bfloat16 \
  --num_workers 1 \
  --prefetch_factor 1 \
  --gpu_batch_multiplier 16 \
  --compile
