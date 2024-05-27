#!/bin/bash

IMG=/home/joosep/singularity/onnxruntime_gpu.simg
MODELS=onnxmodels

# CMD_CPU="singularity exec  --env PYTHONPATH=/opt/onnxruntime-openvino/lib/python3.10/site-packages $IMG python3.10 mlpf/timing.py --execution-provider OpenVINOExecutionProvider"
# $CMD_CPU --model $MODELS/test_fp32.onnx --num-threads 1 | tee timing/openvino_fp32.txt
# $CMD_CPU --model $MODELS/test_fp16.onnx --num-threads 1 --input-dtype float16 | tee timing/openvino_fp16.txt

CMD_GPU="singularity exec --nv --env CUDA_VISIBLE_DEVICES=0 --env PYTHONPATH=/opt/onnxruntime-gpu/lib/python3.10/site-packages $IMG python3.10 mlpf/timing.py --execution-provider CUDAExecutionProvider"
$CMD_GPU --model $MODELS/test_fp32_unfused.onnx --num-threads 1 | tee timing/gpu_fp32_unfused.txt
$CMD_GPU --model $MODELS/test_fp32_fused.onnx --num-threads 1 | tee timing/gpu_fp32_fused.txt

CMD_CPU="singularity exec $IMG python3.10 mlpf/timing.py --execution-provider CPUExecutionProvider"
$CMD_CPU --model $MODELS/test_fp32_unfused.onnx --num-threads 1 | tee timing/cpu_fp32_unfused.txt
$CMD_CPU --model $MODELS/test_fp32_fused.onnx --num-threads 1 | tee timing/cpu_fp32_fused.txt

# CMD_CPU="singularity exec  --env PYTHONPATH=/opt/onnxruntime-openvino/lib/python3.10/site-packages $IMG python3.10 mlpf/timing.py --execution-provider OpenVINOExecutionProvider"
# $CMD_CPU --model $MODELS/test_fp32.onnx --num-threads 1 | tee timing/openvino_fp32.txt

# CMD_GPU="singularity exec --nv --env CUDA_VISIBLE_DEVICES=0 --env LD_LIBRARY_PATH=/home/joosep/particleflow/onnxruntime/build/Linux/Debug/ --env PYTHONPATH=/home/joosep/particleflow/onnxruntime/build/Linux/Debug/ $IMG python3.10 mlpf/timing.py --execution-provider CUDAExecutionProvider"
