#!/bin/bash
unset ROCR_VISIBLE_DEVICES
$WITH_CONDA
source particleflow-env/bin/activate
pip list installed
rocm-smi --showdriverversion
python3 mlpf/pipeline.py \
    --data-dir $TFDS_DATA_DIR \
    --config parameters/pytorch/pyg-cms.yaml \
    --experiment-dir experiments/pyg-cms_20250918_072337_217846 \
    train \
    --gpus 8 \
    --gpu-batch-multiplier 4 \
    --num-workers 1 \
    --prefetch-factor 1 \
    --conv-type attention \
    --dtype bfloat16 \
    --optimizer lamb \
    --lr 0.002 \
    --num-steps 100000 \
    --val-freq 10000 \
    --checkpoint-freq 100 \
    --test-datasets cms_pf_qcd \
    --load experiments/pyg-cms_20250918_072337_217846/checkpoints/checkpoint-49900.pth
