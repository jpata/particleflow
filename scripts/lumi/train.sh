#!/bin/bash
unset ROCR_VISIBLE_DEVICES
$WITH_CONDA
source particleflow-env/bin/activate
pip list installed
rocm-smi --showdriverversion
python3 mlpf/pipeline.py \
    --data-dir $TFDS_DATA_DIR \
    --config parameters/pytorch/pyg-cms.yaml \
    --experiment-dir experiments/pyg-cms_20250912_201225_216787 \
    train \
    --gpus 8 \
    --gpu-batch-multiplier 8 \
    --num-workers 1 \
    --prefetch-factor 1 \
    --conv-type attention \
    --dtype bfloat16 \
    --optimizer lamb \
    --lr 0.002 \
    --num-steps 50000 \
    --val-freq 5000 \
    --checkpoint-freq 100 \
    --test-datasets cms_pf_qcd \
    --load experiments/pyg-cms_20250912_201225_216787/checkpoints/checkpoint-35000.pth
