#!/bin/bash

IMG=/home/joosep/HEP-KBFI/singularity/pytorch.simg
singularity exec $IMG python3.10 mlpf/timing.py --model ~/Dropbox/onnx/test_fp32.onnx --num-threads 1 | tee timing/cpu_fp32.txt
singularity exec $IMG python3.10 mlpf/timing.py --model ~/Dropbox/onnx/test_fp16.onnx --num-threads 1 --input-dtype float16 | tee timing/cpu_fp16.txt
singularity exec $IMG python3.10 mlpf/timing.py --model ~/Dropbox/onnx/test_int8.onnx --num-threads 1 | tee timing/cpu_int8.txt

IMG=/home/joosep/HEP-KBFI/singularity/onnxruntime_gpu.simg
singularity exec --nv --env CUDA_VISIBLE_DEVICES=0 $IMG python3.10 mlpf/timing.py --use-gpu --model ~/Dropbox/onnx/test_fp32.onnx --num-threads 1 | tee timing/gpu_fp32.txt
singularity exec --nv --env CUDA_VISIBLE_DEVICES=0 $IMG python3.10 mlpf/timing.py --use-gpu --model ~/Dropbox/onnx/test_fp16.onnx --num-threads 1 --input-dtype float16 | tee timing/gpu_fp16.txt
singularity exec --nv --env CUDA_VISIBLE_DEVICES=0 $IMG python3.10 mlpf/timing.py --use-gpu --model ~/Dropbox/onnx/test_int8.onnx --num-threads 1 | tee timing/gpu_int8.txt
