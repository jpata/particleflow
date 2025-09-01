#!/bin/bash
unset ROCR_VISIBLE_DEVICES
$WITH_CONDA
source particleflow-env/bin/activate
pip list installed
rocm-smi --showdriverversion
python3 mlpf/pipeline.py \
    --data-dir $TFDS_DATA_DIR \
    --config parameters/pytorch/pyg-cms.yaml \
    train \
    --gpus 8 \
    --gpu-batch-multiplier 16 \
    --num-workers 4 \
    --prefetch-factor 5 \
    --conv-type attention \
    --dtype bfloat16 \
    --optimizer lamb \
    --lr 0.004 \
    --val-freq 1000 \
    --checkpoint-freq 100 \
    --test-datasets cms_pf_qcd \
    --comet
