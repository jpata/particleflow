#!/bin/bash

IMG=/home/software/singularity/pytorch.simg:2024-12-03
MODELS=onnxmodels

CMD_GPU="singularity exec --nv --env CUDA_VISIBLE_DEVICES=0 --env PYTHONPATH=/opt/onnxruntime-gpu/lib/python3.11/site-packages $IMG python3.11 mlpf/timing.py --execution-provider CUDAExecutionProvider"
$CMD_GPU --model $MODELS/test_fp32_unfused.onnx --num-threads 1 | tee timing/gpu_fp32_unfused.txt
$CMD_GPU --model $MODELS/test_fp32_fused.onnx --num-threads 1 | tee timing/gpu_fp32_fused.txt

CMD_CPU="singularity exec $IMG python3.11 mlpf/timing.py --execution-provider CPUExecutionProvider"
$CMD_CPU --model $MODELS/test_fp32_unfused.onnx --num-threads 1 | tee timing/cpu_fp32_unfused.txt
$CMD_CPU --model $MODELS/test_fp32_fused.onnx --num-threads 1 | tee timing/cpu_fp32_fused.txt
