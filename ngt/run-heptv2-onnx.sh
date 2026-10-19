#!/usr/bin/env bash
# Run inside a configured NGT GPU pod, using a shared reference bundle.
set -euo pipefail
if [[ $# != 3 || ! "$1" =~ ^(cuda|migraphx)$ ]]; then
    echo "Usage: PYTHON_BIN=/path/to/python bash ngt/run-heptv2-onnx.sh {cuda|migraphx} BUNDLE OUTDIR" >&2
    exit 2
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $1 == migraphx ]]; then
    # MIGraphX otherwise narrows the exact integer hash arithmetic. Set this
    # before loading its libraries; the compiler caches environment settings.
    case ",${MIGRAPHX_DISABLE_PASSES:-}," in
        *,eliminate_data_type,*) ;;
        *) export MIGRAPHX_DISABLE_PASSES="${MIGRAPHX_DISABLE_PASSES:+${MIGRAPHX_DISABLE_PASSES},}eliminate_data_type" ;;
    esac
    export MIGRAPHX_GPU_COMPILE_PARALLEL=${MIGRAPHX_GPU_COMPILE_PARALLEL:-4}
    # LLVM's GVN memory-dependence analysis is very slow on full int64 sort
    # kernels above 4096 elements. Keep O3 and disable that specific analysis.
    export MIGRAPHX_GPU_OPTIMIZE=${MIGRAPHX_GPU_OPTIMIZE:-3}
    case " ${MIGRAPHX_GPU_HIP_FLAGS:-} " in
        *" -enable-gvn-memdep=false "*) ;;
        *) export MIGRAPHX_GPU_HIP_FLAGS="${MIGRAPHX_GPU_HIP_FLAGS:+${MIGRAPHX_GPU_HIP_FLAGS} }-mllvm -enable-gvn-memdep=false" ;;
    esac
fi
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python3}" "$repo_dir/scripts/qualify-heptv2-onnx.py" run \
    --provider "$1" --bundle "$2" --outdir "$3" --static-shapes
