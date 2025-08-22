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
    --prefetch-factor 10 \
    --checkpoint-freq 1 \
    --conv-type attention \
    --dtype bfloat16 \
    --optimizer lamb \
    --lr 0.0005 \
    --ntest 1000 \
    --test-datasets cms_pf_qcd
