#!/bin/bash

export CMD="singularity exec --env CUDA_VISIBLE_DEVICES=0 --nv /home/joosep/HEP-KBFI/singularity/tf-2.13.0rc2.simg python3 mlpf/pipeline.py train"

$CMD --config parameters/mixedprecision/clic_fp32_bs1.yaml --ntrain 5000 --ntest 5000 --nepochs 10
$CMD --config parameters/mixedprecision/clic_bf16_bs1.yaml --ntrain 5000 --ntest 5000 --nepochs 10
$CMD --config parameters/mixedprecision/clic_bf16_bs2.yaml --ntrain 5000 --ntest 5000 --nepochs 10
$CMD --config parameters/mixedprecision/clic_fp16_bs1.yaml --ntrain 5000 --ntest 5000 --nepochs 10
$CMD --config parameters/mixedprecision/clic_fp16_bs2.yaml --ntrain 5000 --ntest 5000 --nepochs 10

$CMD --config parameters/mixedprecision/clic_fp32_bs1.yaml --ntrain 5000 --ntest 5000 --nepochs 10
$CMD --config parameters/mixedprecision/clic_bf16_bs1.yaml --ntrain 5000 --ntest 5000 --nepochs 10
$CMD --config parameters/mixedprecision/clic_bf16_bs2.yaml --ntrain 5000 --ntest 5000 --nepochs 10
$CMD --config parameters/mixedprecision/clic_fp16_bs1.yaml --ntrain 5000 --ntest 5000 --nepochs 10
$CMD --config parameters/mixedprecision/clic_fp16_bs2.yaml --ntrain 5000 --ntest 5000 --nepochs 10
