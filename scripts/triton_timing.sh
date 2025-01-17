#!/bin/bash

# Ensure this file model_repository/mlpf/1/model.onnx exists and points to the latest model from huggingface.
# Then run the server: singularity exec --nv docker://nvcr.io/nvidia/tritonserver:24.06-py3 tritonserver --model-repository model_repository

# Now run the inference performance test on different event sizes.
rm -f timing/gpu_triton.txt
for bin in 2560 5120 10240; do
    echo "number of elements: " $bin >> timing/gpu_triton.txt
    perf_analyzer -m mlpf --percentile=95 --shape Xfeat_normed:$bin,55 --shape mask:$bin -b 4 >> timing/gpu_triton.txt
done
