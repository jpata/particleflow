#!/usr/bin/env bash
set -euo pipefail
cd /tmp/particleflow
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/opt/rocm-7.2.4/bin:$PATH
export LD_LIBRARY_PATH=/tmp/sdpa-cxx-runtime/lib:/opt/rocm-7.2.4/lib:/opt/rocm-7.2.4/lib64
export STUDY_PYTHON=/tmp/sdpa-w7900-venv/bin/python
root=results/attention-study
while [[ ! -f "$root/rocm-attention/edge-correctness.json" ]]; do sleep 5; done
bash scripts/attention-study/diagnostics.sh migraphx > /tmp/study-rocm-diagnostics.log 2>&1
touch "$root/rocm-followup-complete"
