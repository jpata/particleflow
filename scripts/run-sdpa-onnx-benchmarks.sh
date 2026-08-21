#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_SCRIPT="${ROOT_DIR}/scripts/benchmark-sdpa-onnx.py"
PLOT_SCRIPT="${ROOT_DIR}/scripts/plot-sdpa-onnx-summary.py"

OUTDIR="${OUTDIR:-${ROOT_DIR}/sdpa_onnx_bench}"
SEQ_LENS="${SEQ_LENS:-1280 2560 5120 10240}"
#SEQ_LENS="${SEQ_LENS:-128 256 512 1024}"
CPU_ITERS="${CPU_ITERS:-5}"
GPU_ITERS="${GPU_ITERS:-5}"
WARMUP="${WARMUP:-3}"
GPU_WARMUP="${GPU_WARMUP:-3}"
NUM_THREADS="${NUM_THREADS:-1}"
MEMORY_SAMPLE_INTERVAL="${MEMORY_SAMPLE_INTERVAL:-0.02}"
NUM_HEADS="${NUM_HEADS:-16}"
HEAD_DIM="${HEAD_DIM:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODES="${MODES:-both}"
KEEP_MODELS="${KEEP_MODELS:-0}"
PROFILE_GPU="${PROFILE_GPU:-0}"
RUN_CPU="${RUN_CPU:-1}"
RUN_GPU="${RUN_GPU:-1}"

BENCH_ARGS=(
  --seq-lens ${SEQ_LENS}
  --batch-size "${BATCH_SIZE}"
  --num-heads "${NUM_HEADS}"
  --head-dim "${HEAD_DIM}"
  --num-threads "${NUM_THREADS}"
  --memory-sample-interval "${MEMORY_SAMPLE_INTERVAL}"
  --modes "${MODES}"
)

if [[ "${KEEP_MODELS}" == "1" ]]; then
  BENCH_ARGS+=(--keep-models)
fi

print_config() {
  cat <<EOF
SDPA ONNX benchmark configuration
  output:      ${OUTDIR}
  seq_lens:    ${SEQ_LENS}
  shape:       batch=${BATCH_SIZE}, heads=${NUM_HEADS}, head_dim=${HEAD_DIM}
  modes:       ${MODES}
  threads:     ${NUM_THREADS}
  mem sample:  ${MEMORY_SAMPLE_INTERVAL}s
  cpu:         run=${RUN_CPU}, iters=${CPU_ITERS}, warmup=${WARMUP}
  gpu:         run=${RUN_GPU}, iters=${GPU_ITERS}, warmup=${GPU_WARMUP}, profile=${PROFILE_GPU}
EOF
}

gpu_available() {
  uv run --project "${ROOT_DIR}/envs/ort-gpu" python3 - <<'PY'
import sys
import torch
import onnxruntime as ort

ok = torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers()
if ok:
    print(torch.cuda.get_device_name(0))
    sys.exit(0)
print("CUDAExecutionProvider is not available")
sys.exit(1)
PY
}

run_benchmark() {
  local label="$1"
  local project="$2"
  local device="$3"
  local iters="$4"
  local warmup="$5"
  local outdir="$6"
  shift 6

  echo
  echo "Running ${label} benchmark..."
  uv run --project "${ROOT_DIR}/${project}" python3 "${BENCH_SCRIPT}" \
    --device "${device}" \
    --iters "${iters}" \
    --warmup "${warmup}" \
    --outdir "${outdir}" \
    "${BENCH_ARGS[@]}" \
    "$@"
}

summarize_csv() {
  local title="$1"
  local csv_path="$2"
  if [[ ! -f "${csv_path}" ]]; then
    echo
    echo "${title}: no summary found at ${csv_path}"
    return
  fi

  echo
  echo "${title}"
  uv run --project "${ROOT_DIR}/envs/ort-cpu" python3 - "${csv_path}" <<'PY'
import csv
import math
import sys
from collections import defaultdict

path = sys.argv[1]
rows = []
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

ok = []
failed = []
for row in rows:
    if row["status"] == "ok" and row.get("mean_ms"):
        row["sequence_length"] = int(row["sequence_length"])
        row["mean_ms"] = float(row["mean_ms"])
        row["p90_ms"] = float(row["p90_ms"])
        row["peak_memory_mib"] = float(row["peak_memory_mib"]) if row.get("peak_memory_mib") else math.nan
        row["mean_peak_memory_mib"] = float(row["mean_peak_memory_mib"]) if row.get("mean_peak_memory_mib") else math.nan
        row["mae"] = float(row["mae"]) if row.get("mae") else math.nan
        ok.append(row)
    else:
        failed.append(row)

by_size = defaultdict(list)
for row in ok:
    by_size[row["sequence_length"]].append(row)

for seq_len in sorted(by_size):
    ranked = sorted(by_size[seq_len], key=lambda r: r["mean_ms"])
    print(f"\nS={seq_len}")
    print(f"{'rank':>4}  {'variant':<34} {'mean ms':>10} {'p90 ms':>10} {'peak MiB':>10} {'MAE':>11}")
    for idx, row in enumerate(ranked, 1):
        print(
            f"{idx:4d}  {row['variant']:<34} "
            f"{row['mean_ms']:10.3f} {row['p90_ms']:10.3f} "
            f"{row['peak_memory_mib']:10.1f} {row['mae']:11.3e}"
        )

if failed:
    print("\nFailures / unsupported variants")
    print(f"{'seq':>6}  {'variant':<34} {'error'}")
    for row in failed:
        print(f"{row['sequence_length']:>6}  {row['variant']:<34} {row.get('error', '')}")
PY
}

plot_summary() {
  local cpu_csv="${OUTDIR}/cpu/summary.csv"
  local gpu_csv="${OUTDIR}/gpu/summary.csv"
  if [[ ! -f "${cpu_csv}" && ! -f "${gpu_csv}" ]]; then
    echo
    echo "Skipping summary plot: no CPU or GPU CSV summaries found"
    return
  fi

  echo
  echo "Creating summary plot..."
  uv run --project "${ROOT_DIR}/envs/ort-cpu" python3 "${PLOT_SCRIPT}" \
    --cpu-csv "${cpu_csv}" \
    --gpu-csv "${gpu_csv}" \
    --outdir "${OUTDIR}/plots"
}

main() {
  print_config
  mkdir -p "${OUTDIR}"

  if [[ "${RUN_CPU}" == "1" ]]; then
    run_benchmark "CPU" "envs/ort-cpu" "cpu" "${CPU_ITERS}" "${WARMUP}" "${OUTDIR}/cpu"
  fi

  if [[ "${RUN_GPU}" == "1" ]]; then
    if gpu_name="$(gpu_available)"; then
      echo
      echo "GPU detected: ${gpu_name}"
      gpu_extra=()
      if [[ "${PROFILE_GPU}" == "1" ]]; then
        gpu_extra+=(--profile)
      fi
      run_benchmark "GPU" "envs/ort-gpu" "cuda" "${GPU_ITERS}" "${GPU_WARMUP}" "${OUTDIR}/gpu" "${gpu_extra[@]}"
    else
      echo
      echo "Skipping GPU benchmark: ${gpu_name}"
    fi
  fi

  summarize_csv "CPU summary" "${OUTDIR}/cpu/summary.csv"
  summarize_csv "GPU summary" "${OUTDIR}/gpu/summary.csv"
  plot_summary

  echo
  echo "Full outputs:"
  echo "  ${OUTDIR}/cpu/summary.json"
  echo "  ${OUTDIR}/cpu/summary.csv"
  echo "  ${OUTDIR}/gpu/summary.json"
  echo "  ${OUTDIR}/gpu/summary.csv"
  echo "  ${OUTDIR}/plots/sdpa_onnx_summary.png"
  echo "  ${OUTDIR}/plots/sdpa_onnx_summary.pdf"
}

main "$@"
