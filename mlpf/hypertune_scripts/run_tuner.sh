#!/bin/sh

export KERASTUNER_TUNER_ID=$1
export KERASTUNER_ORACLE_IP=$2
export KERASTUNER_ORACLE_PORT=$3
echo "KERASTUNER_TUNER_ID:"
echo $KERASTUNER_TUNER_ID
echo "KERASTUNER_ORACLE_IP:"
echo $KERASTUNER_ORACLE_IP
echo "KERASTUNER_ORACLE_PORT:"
echo $KERASTUNER_ORACLE_PORT


nvidia-smi
echo 'Starting tuner.'
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py hypertune -c parameters/cms-gnn-dense-short.yaml -o $4
echo 'Tuner done.'
