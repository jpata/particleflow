#!/bin/bash
#SBATCH --job-name=colliderml_tfds
#SBATCH --partition=genx
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs_slurm/colliderml_tfds_%A_%a.out
#SBATCH --error=logs_slurm/colliderml_tfds_%A_%a.err
#SBATCH --export=NONE

set -euo pipefail

# TFDS build for ColliderML — the single entry point. The dataset has 10 TFDS configs
# (NUM_SPLITS in mlpf/heptfds/colliderml_utils/utils.py); each is an independently
# prepared dataset.
#
# Usage:
#   # parallel: one array task per config of colliderml_ttbar_nopu_pf:
#   sbatch --array=1-10 scripts/flatiron/colliderml_tfds.sh
#
#   # sequential: all 10 configs in one job:
#   sbatch scripts/flatiron/colliderml_tfds.sh
#
#   # pu200 (colliderml_ttbar_pu200_pf):
#   sbatch --array=1-10 scripts/flatiron/colliderml_tfds.sh pu200
#
#   # any other Release-1 sample <process>_<pileup>: builder colliderml_pf/<process>.py for pu0
#   # (dataset colliderml_<process>_nopu_pf), colliderml_pf/<process>_pu200.py for pu200:
#   sbatch --array=1-10 scripts/flatiron/colliderml_tfds.sh zee_pu200
#
#   # rebuilt parquet in a custom dir (2nd positional arg; --export=NONE means env
#   # vars don't propagate):
#   sbatch --array=1-10 scripts/flatiron/colliderml_tfds.sh pu0 /path/to/mlpf_parquet
#
# Safe to parallelize: split_sample() gives each config a disjoint slice of the manual_dir
# parquet shards, and each config prepares into its own directory under --data_dir (TFDS
# preparation is atomic per config: tmp dir + rename; an already-prepared config is
# skipped). Re-running after a converter change requires either deleting the prepared
# <data_dir>/<dataset>/<config>/<version>/ tree first or bumping the builder VERSION.

SAMPLE="${1:-pu0}"
# a bare pileup label is the ttbar sample (the original usage); otherwise <process>_<pileup>
case "${SAMPLE}" in
    pu0|pu200) SAMPLE="ttbar_${SAMPLE}" ;;
esac
PROCESS="${SAMPLE%_*}"
PU="${SAMPLE##*_}"
MANUAL_DIR="${2:-/mnt/ceph/users/ewulff/data/colliderml/mlpf_parquet/clue/${SAMPLE}}"
TFDS_DATA_DIR="/mnt/ceph/users/ewulff/tensorflow_datasets/colliderml"

case "${PU}" in
    pu0) BUILDER="mlpf/heptfds/colliderml_pf/${PROCESS}" ;;
    pu200) BUILDER="mlpf/heptfds/colliderml_pf/${PROCESS}_pu200" ;;
    *) echo "ERROR: unknown pileup label '${PU}' in sample ${SAMPLE} (known: pu0, pu200)" >&2; exit 2 ;;
esac
if [ ! -f "${BUILDER}.py" ]; then
    echo "ERROR: no TFDS builder ${BUILDER}.py for sample ${SAMPLE}" >&2
    exit 2
fi

# array mode: one config per task; non-array: all configs sequentially in one job
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "${SLURM_ARRAY_TASK_ID}" -lt 1 ] || [ "${SLURM_ARRAY_TASK_ID}" -gt 10 ]; then
        echo "ERROR: array task id ${SLURM_ARRAY_TASK_ID} out of range; TFDS configs are 1..10 (use --array=1-10)" >&2
        exit 2
    fi
    CONFIGS=("${SLURM_ARRAY_TASK_ID}")
else
    CONFIGS=(1 2 3 4 5 6 7 8 9 10)
fi

# the tfds CLI lives in this repo's uv environment (no torch needed: the builder only
# imports mlpf.conf); submitted from the repo root, .venv resolves relative to $PWD
if [ ! -x .venv/bin/tfds ]; then
    echo "ERROR: .venv/bin/tfds not found in $(pwd); submit from the repo root and run 'uv sync' first" >&2
    exit 2
fi
export PYTHONPATH="$(pwd)"

# completeness check: the 90/10 train/test split needs >=10 test files for 10 configs, so
# ~100+ parquet shards are required for the build to succeed at all. Nothing in the build
# verifies that ALL converted shards are present (a partially converted manual_dir with
# >=100 shards yields a silently truncated dataset) — check the count matches your intent.
# (a nullglob array, not `ls | wc -l`: under pipefail a missing dir made that exit silently)
shopt -s nullglob
PARQUET_FILES=("${MANUAL_DIR}"/*.parquet)
shopt -u nullglob
N_PARQUET="${#PARQUET_FILES[@]}"
echo "manual_dir ${MANUAL_DIR}: ${N_PARQUET} parquet shards"
if [ "${N_PARQUET}" -lt 100 ]; then
    echo "ERROR: need >=100 converted shards for the 10-config/90-10 split, found ${N_PARQUET}." >&2
    echo "  finish conversion first: sbatch scripts/flatiron/colliderml_convert.sh 0 1000 ${SAMPLE}" >&2
    exit 2
fi

echo "building ${BUILDER} config(s) ${CONFIGS[*]} on $(hostname): manual_dir=${MANUAL_DIR} -> ${TFDS_DATA_DIR}"

for CONFIG in "${CONFIGS[@]}"; do
    .venv/bin/tfds build "${BUILDER}" \
        --config "${CONFIG}" \
        --manual_dir "${MANUAL_DIR}" \
        --data_dir "${TFDS_DATA_DIR}" \
        --beam_pipeline_options "direct_num_workers=${SLURM_CPUS_PER_TASK:-1}"
done

echo "done"
