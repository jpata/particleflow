#!/usr/bin/env bash
# Run inside the pod from /tmp/particleflow after session.sh copy.
set -euo pipefail
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export LD_LIBRARY_PATH="/tmp/sdpa-cxx-runtime/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source /tmp/sdpa-w7900-venv/bin/activate
outdir=${1:-results/pr523/w7900}
common=(--device cuda --ort-provider migraphx --require-full-gpu
    --mask-fraction 0.5 --num-heads 16 --head-dim 16 --warmup 5 --iters 30
    --variants pt_flash_fp16 ort_attention24_fp16 ort_mha_fp16
    --export-shapes both --keep-models)
python scripts/benchmark-sdpa-onnx.py "${common[@]}" \
    --batch-sizes 1 2 4 8 16 --seq-lens 512 1280 \
    --length-policy balanced --modes masked --outdir "$outdir/batch"
python scripts/benchmark-sdpa-onnx.py "${common[@]}" \
    --batch-size 4 --seq-lens 128 512 1280 2560 \
    --modes both --profile --outdir "$outdir/sequence"
