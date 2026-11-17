#!/bin/bash
set -euo pipefail
REPO_ROOT=${MLPF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
source "$REPO_ROOT/scripts/empireai/environment.sh"
require_arm_environment
cd "$REPO_ROOT"
SCENARIO_DIR=${1:-experiments/cld_gh200_backbone_benchmark}
python3 scripts/empireai/summarize_benchmark.py "$SCENARIO_DIR"
shopt -s globstar nullglob
SUMMARIES=("$SCENARIO_DIR"/**/benchmark/*/summary.json)
if (( ${#SUMMARIES[@]} )); then
  python3 -m mlpf.benchmark_report "$SCENARIO_DIR"
fi
