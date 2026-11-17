#!/usr/bin/env bash
# Inside the configured W7900 pod, from /tmp/particleflow.
set -euo pipefail
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="/tmp/sdpa-cxx-runtime/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source /tmp/sdpa-w7900-venv/bin/activate
outdir=${1:-results/math-w7900}
python scripts/benchmark-sdpa-onnx.py \
    --device cuda --ort-provider migraphx --require-full-gpu \
    --batch-size 4 --seq-lens 128 512 1280 2560 --modes both \
    --mask-fraction 0.5 --num-heads 16 --head-dim 16 --warmup 5 --iters 30 \
    --variants pt_math_fp32 pt_math_fp16 onnx_sdpa_export_fp32 onnx_sdpa_export_fp16 \
    --export-shapes static --keep-models --profile --outdir "$outdir"
mkdir -p "$outdir/kernels"
for dtype in fp32 fp16; do
    MIGRAPHX_TRACE_COMPILE=1 rocprofv3 --kernel-trace --stats --output-format csv \
        --output-directory "$outdir/kernels/$dtype" -- \
        python ngt/probe-ort-kernels.py \
        "$outdir/models/batch_4/onnx_sdpa_export_${dtype}_static_unmasked_s512.onnx" \
        > "$outdir/kernels/$dtype.log" 2>&1
done
