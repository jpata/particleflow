#!/usr/bin/env bash
# Run from /tmp/particleflow after timing sweeps have finished.
set -euo pipefail
backend=${1:?cuda or migraphx}
python=${STUDY_PYTHON:?}
root=results/attention-study
mkdir -p "$root/diagnostics" "$root/memory"
probe() { "$python" scripts/attention-study/probe.py "$@"; }
if [[ $backend == cuda ]]; then
    for item in sdpa_legacy_fp16_unmasked attention24_dense_fp16_unmasked attention24_dense_fp16_padding attention24_lengths_fp16_padding; do
        out="$root/diagnostics/cuda-$item"
        mkdir -p "$out"
        nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true \
            --output="$out/trace" "$python" scripts/attention-study/probe.py \
            "$root/bundle/${item}_b4_s512.onnx" --provider cuda --out "$out" > "$out/nsys.log" 2>&1
        nsys stats --report cuda_gpu_kern_sum --format csv "$out/trace.nsys-rep" > "$out/kernels.csv" 2> "$out/stats.log"
    done
    probe "$root/bundle/sdpa_legacy_fp16_unmasked_b4_s512_dynamic.onnx" --provider cuda --allow-cpu --sizes 512 1280 512 2560 512 --out "$root/diagnostics/cuda-dynamic" > "$root/diagnostics/cuda-dynamic.log" 2>&1
else
    for setting in default attention; do
        if [[ $setting == attention ]]; then export MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention; else unset MIGRAPHX_MLIR_USE_SPECIFIC_OPS; fi
        for item in sdpa_legacy_fp16_unmasked explicit_safe_fp16_padding; do
            out="$root/diagnostics/rocm-$setting-$item"
            mkdir -p "$out"
            MIGRAPHX_TRACE_COMPILE=1 rocprofv3 --kernel-trace --stats --output-format csv --output-directory "$out/rocprof" -- \
                "$python" scripts/attention-study/probe.py "$root/bundle/${item}_b4_s512.onnx" --provider migraphx --out "$out" > "$out/compile.log" 2>&1
        done
    done
    export MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention
    probe "$root/bundle/sdpa_legacy_fp16_padding_b4_s512.onnx" --provider migraphx --allow-cpu --out "$root/diagnostics/rocm-mask-fallback" > "$root/diagnostics/rocm-mask-fallback.log" 2>&1
    for cache in none disk; do
        extra=()
        if [[ $cache == disk ]]; then
            # This scenario produced a GPU memory fault in the recorded environment.
            # Repeating it is a deliberate diagnostic, not part of the default run.
            if [[ ${STUDY_TEST_DISK_CACHE:-0} != 1 ]]; then continue; fi
            extra=(--cache "$root/diagnostics/compiled-cache")
        fi
        if probe "$root/bundle/sdpa_legacy_fp16_unmasked_b4_s512_dynamic.onnx" --provider migraphx --allow-cpu \
            --sizes 512 1280 512 2560 512 "${extra[@]}" --out "$root/diagnostics/rocm-dynamic-$cache" > "$root/diagnostics/rocm-dynamic-$cache.log" 2>&1; then
            :
        else
            status=$?
            printf '{"exit_code": %s}\n' "$status" > "$root/diagnostics/rocm-dynamic-$cache.failure.json"
        fi
    done
fi
for seq in 512 2560; do
    for item in sdpa_legacy_fp16_unmasked explicit_safe_fp16_padding; do
        "$python" scripts/attention-study/memory.py "$root/bundle/${item}_b4_s${seq}.onnx" --provider "$backend" \
            --out "$root/memory/${backend}-${item}-s${seq}.json"
    done
done
