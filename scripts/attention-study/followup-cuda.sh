#!/usr/bin/env bash
set -euo pipefail
cd /tmp/particleflow
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export STUDY_PYTHON=/tmp/attention-venv/bin/python
root=results/attention-study
while [[ ! -f "$root/cpu123/edge-correctness.json" ]]; do sleep 5; done
cp -a "$root/cuda129" "$root/cuda129-before-binding-fix"
"$STUDY_PYTHON" -u scripts/attention-study/study.py run --bundle "$root/bundle" --out "$root/cuda129" --provider cuda --iters 10 --warmup 5 > /tmp/study-cuda129-final.log 2>&1
bash scripts/attention-study/diagnostics.sh cuda > /tmp/study-cuda-diagnostics.log 2>&1
"$STUDY_PYTHON" scripts/attention-study/batch_bundle.py "$root/batch-bundle" > /tmp/study-batch-export.log 2>&1
"$STUDY_PYTHON" -u scripts/attention-study/study.py run --bundle "$root/batch-bundle" --out "$root/cuda-batch" --provider cuda --iters 10 --warmup 5 > /tmp/study-cuda-batch.log 2>&1
"$STUDY_PYTHON" -u scripts/attention-study/study.py run --bundle "$root/batch-bundle" --out "$root/cpu-batch" --provider cpu --iters 10 --warmup 5 > /tmp/study-cpu-batch.log 2>&1
"$STUDY_PYTHON" scripts/attention-study/override_demo.py --wrapper scripts/attention-study/mlpf-onnx-attention.py --out "$root/override-demo" > /tmp/study-override.log 2>&1
for seq in 512 2560; do
 "$STUDY_PYTHON" scripts/attention-study/memory.py "$root/bundle/sdpa_legacy_fp16_unmasked_b4_s${seq}.onnx" --provider cpu --out "$root/memory/cpu-legacy-s${seq}.json"
done
touch "$root/cuda-followup-complete"
