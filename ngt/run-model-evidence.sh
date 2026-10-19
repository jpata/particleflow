#!/usr/bin/env bash
# Run inside the configured runtime with PYTHONPATH and AMD library paths set.
set -uo pipefail
provider=${1:?cpu, cuda or migraphx}
root=${2:?bundle root containing attention/ and heptv2/}
out=${3:?result directory}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${PYTHON_BIN:-python3}
models=("${@:4}")
if [[ ${#models[@]} == 0 ]]; then models=(attention heptv2); fi
if [[ $provider == migraphx ]]; then
    export MIGRAPHX_DISABLE_PASSES=${MIGRAPHX_DISABLE_PASSES-eliminate_data_type}
    export MIGRAPHX_GPU_COMPILE_PARALLEL=4
    export MIGRAPHX_GPU_OPTIMIZE=3
    export MIGRAPHX_GPU_HIP_FLAGS=${MIGRAPHX_GPU_HIP_FLAGS-'-mllvm -enable-gvn-memdep=false'}
fi
mkdir -p "$out"
status=0
for model in "${models[@]}"; do
    for case in "$root/$model"/sweep_*.npz; do
        name=$(basename "$case" .npz)
        for mode in timing placement; do
            dest="$out/$model/$name/$mode"
            mkdir -p "$dest"
            args=(--warmup 10 --iters 100)
            if [[ $mode == placement ]]; then args=(--placement --warmup 0 --iters 1); fi
            if [[ ${CUDA_FLASH:-0} == 1 ]]; then args+=(--flash); fi
            worker_env=()
            if [[ $mode == placement && ${TRACE_COMPILE_PLACEMENT:-0} == 1 ]]; then
                worker_env=(env MIGRAPHX_TRACE_COMPILE=1)
            fi
            "${worker_env[@]}" timeout 1800 "$python" "$repo/scripts/benchmark-model-export.py" \
                --bundle "$root/$model" --case "$(basename "$case")" \
                --provider "$provider" --outdir "$dest" "${args[@]}" > "$dest/run.log" 2>&1
            code=$?
            echo "$model $name $mode exit=$code"
            echo "$code" > "$dest/exit-code.txt"
            if [[ $code != 0 ]]; then status=1; fi
        done
    done
done
exit "$status"
