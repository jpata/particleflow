#!/bin/bash
export PF_SITE=local
# Use /mnt path which is bound in the container
DATA_DIR=/mnt/work/mlpf/cms/20260204_cmssw_15_0_5_117d32/tfds/
./scripts/local/wrapper.sh python3 mlpf/standalone/eval.py --data-dir $DATA_DIR --attention-type fastformer > log1.txt
./scripts/local/wrapper.sh python3 mlpf/standalone/eval.py --data-dir $DATA_DIR --attention-type global > log2.txt 
./scripts/local/wrapper.sh python3 mlpf/standalone/eval.py --data-dir $DATA_DIR --attention-type hept > log3.txt 
./scripts/local/wrapper.sh python3 mlpf/standalone/eval.py --data-dir $DATA_DIR --attention-type standard > log4.txt
