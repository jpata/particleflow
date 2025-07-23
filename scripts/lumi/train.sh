#!/bin/bash
unset ROCR_VISIBLE_DEVICES
$WITH_CONDA
source particleflow-env/bin/activate
pip list installed
rocm-smi --showdriverversion
python3 mlpf/pipeline.py --gpus 8 \
    --data-dir $TFDS_DATA_DIR --config parameters/pytorch/pyg-cms.yaml \
    --train --gpu-batch-multiplier 16 --num-workers 8 --prefetch-factor 10 --checkpoint-freq 1 --conv-type attention --dtype bfloat16 --optimizer lamb --lr 0.005 --ntest 1000 --test-datasets cms_pf_qcd
