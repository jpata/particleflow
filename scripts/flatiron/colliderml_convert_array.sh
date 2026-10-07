#!/bin/bash
#SBATCH --job-name=colliderml_convert_array
#SBATCH --partition=genx
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --output=logs_slurm/colliderml_convert_array_%A_%a.out
#SBATCH --error=logs_slurm/colliderml_convert_array_%A_%a.err
#SBATCH --export=NONE

set -euo pipefail

# Convert a contiguous range of ColliderML shards on Rusty (array-tasks variant; see
# scripts/flatiron/colliderml_convert.sh for the whole-node variant).
# Usage:
#   # convert shards 0..99 sequentially in one process (not recommended — the converter's
#   # RSS grows over shards; a long sequential run can exceed 16 GB):
#   sbatch scripts/flatiron/colliderml_convert_array.sh 0 100
#
#   # convert shards 0..99 as N array tasks, each covering a contiguous slice (any N works;
#   # e.g. --array=0-19 gives 20 tasks of 5 shards each, --array=0-99 gives 100 tasks of 1):
#   sbatch --array=0-19 scripts/flatiron/colliderml_convert_array.sh 0 100
#
#   # same but for the pu200 sample (third arg is the pileup label, default pu0):
#   sbatch --array=0-19 scripts/flatiron/colliderml_convert_array.sh 0 100 pu200
#
#   # custom output dir (4th positional arg; --export=NONE means env vars don't propagate):
#   sbatch --array=0-19 scripts/flatiron/colliderml_convert_array.sh 0 100 pu0 /path/to/outdir
#
# TFDS building is a separate step: scripts/flatiron/colliderml_tfds.sh
#
#   # to bound concurrent array tasks: append %N to the range, e.g. --array=0-99%50
#
# Default: shards 0 to 100, sample ttbar_pu0.
#
# Output:
#   /mnt/ceph/users/ewulff/data/colliderml/mlpf_parquet/clustered/ttbar_${PU}/train-NN-of-01000.parquet
#   (or the 4th positional arg's directory)
S0="${1:-0}"
S1="${2:-100}"
PU="${3:-pu0}"
# Optional 4th positional arg: output dir override. Jobs run with --export=NONE, so pass it
# positionally; COLLIDERML_OUT_DIR below only helps direct/local runs.
OUTDIR_ARG="${4:-}"

# In array mode the range [S0, S1) is split evenly across the tasks: N tasks each take
# ceil(shards/N) shards. SLURM_ARRAY_TASK_{ID,MIN,MAX} come from the scheduler.
RANGE_START="${S0}"
RANGE_END="${S1}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TASK_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
  TASK_MAX="${SLURM_ARRAY_TASK_MAX:-${SLURM_ARRAY_TASK_ID}}"
  N_TASKS=$(( TASK_MAX - TASK_MIN + 1 ))
  N_SHARDS=$(( RANGE_END - RANGE_START ))
  N_CHUNK=$(( (N_SHARDS + N_TASKS - 1) / N_TASKS ))
  S0=$(( RANGE_START + (SLURM_ARRAY_TASK_ID - TASK_MIN) * N_CHUNK ))
  S1=$(( S0 + N_CHUNK ))
  if [ "$S1" -gt "$RANGE_END" ]; then S1="$RANGE_END"; fi
  if [ "$S0" -ge "$RANGE_END" ]; then
    echo "array task ${SLURM_ARRAY_TASK_ID} is beyond the shard range; nothing to do"
    exit 0
  fi
fi
echo "converting shards [${S0}, ${S1}) on $(hostname)${SLURM_ARRAY_TASK_ID:+ (array task ${SLURM_ARRAY_TASK_ID})}"

PYTHON=/mnt/home/ewulff/repositories/particleflow/.venv/bin/python
export PYTHONPATH="/mnt/home/ewulff/repositories/particleflow"
export KERAS_BACKEND=torch

SAMPLE="ttbar_${PU}"
SOURCE_DIR="/mnt/ceph/users/ewulff/data/colliderml/CERN__ColliderML-Release-1"
# COLLIDERML_OUT_DIR overrides the output root (default: the production clustered dir);
# the optional 4th positional arg wins over it.
OUT_DIR="${OUTDIR_ARG:-${COLLIDERML_OUT_DIR:-/mnt/ceph/users/ewulff/data/colliderml/mlpf_parquet/clustered/${SAMPLE}}}"
mkdir -p "${OUT_DIR}"

# Clustering settings per pileup (tuned 2026-10-05 via radii/merge sweeps on 10 ttbar_pu200
# events from shards 0-1 plus 5 ttbar_pu0 events).
# pu0: bfs_merge with merge_frac=0.25 (the tuned default; fragment merge recovers shower
# tails and there is no pileup to bridge clusters).
# pu200: any fragment merging cascades through the dense pileup-linked graph into
# O(10^5)-hit / multi-TeV mega-clusters (~200k hits / 4.5 TeV largest with the pu0
# settings; transitivity defeats even frozen-threshold/capped merges), so the merge is
# switched off (bfs) and the link radii tightened to ECAL 16.25 mm (0.65x) / HCAL 36 mm
# (0.4x): measured pu200 purity 0.20 -> 0.69 (E-weighted dominant-particle share) and the
# largest cluster drops to ~500 hits / ~70 GeV.
# ALGO / MERGE_FRAC / RADII_* are job-side defaults; to override them, edit this script
# (or switch to args if that becomes common).
if [ "${PU}" = "pu200" ]; then
  ALGO="bfs"
  MERGE_FRAC="0"
  RADII_ECAL="16.25"
  RADII_HCAL="36"
else
  ALGO="bfs_merge"
  MERGE_FRAC="0.25"
  RADII_ECAL="25"
  RADII_HCAL="90"
fi

${PYTHON} -m mlpf.data.colliderml.postprocessing \
    --input "${SOURCE_DIR}" \
    --sample "${SAMPLE}" \
    --outpath "${OUT_DIR}" \
    --shards "${S0}:${S1}" \
    --algorithm "${ALGO}" \
    --merge-frac "${MERGE_FRAC}" \
    --radii-ecal "${RADII_ECAL}" \
    --radii-hcal "${RADII_HCAL}"
echo "done"
