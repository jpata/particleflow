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
#   /mnt/ceph/users/ewulff/data/colliderml/mlpf_parquet/clue/ttbar_${PU}/train-NN-of-01000.parquet
#   (or the 4th positional arg's directory; the pre-CLUE bfs parquet stays in mlpf_parquet/clustered/)
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
# COLLIDERML_OUT_DIR overrides the output root (default: the production CLUE dir);
# the optional 4th positional arg wins over it.
OUT_DIR="${OUTDIR_ARG:-${COLLIDERML_OUT_DIR:-/mnt/ceph/users/ewulff/data/colliderml/mlpf_parquet/clue/${SAMPLE}}}"
mkdir -p "${OUT_DIR}"

# Clustering: CLUE (mlpf/data/colliderml/clue.py) with the per-pileup preset of
# clue.CLUE_PRESETS, as decided 2026-10-08:
# pu0 = the CLUE defaults; pu200 = the "middle setting" (lower seed threshold, smaller
# seed_dc/dm, weaker absorption: ~2.5x the default's clusters at pu200, halving the
# allocator merges of hard-scatter targets at the cost of more split particles).
# Until 2026-10-10 production used bfs_merge (pu0, merge-frac 0.25, radii 25/90 mm) and
# bfs (pu200, merge-frac 0, radii 16.25/36 mm); that parquet is in mlpf_parquet/clustered/.
case "${PU}" in
  pu0|pu200) CLUE_PRESET="${PU}" ;;
  *) echo "no CLUE preset for pileup label '${PU}' (known: pu0, pu200)" >&2; exit 2 ;;
esac

${PYTHON} -m mlpf.data.colliderml.postprocessing \
    --input "${SOURCE_DIR}" \
    --sample "${SAMPLE}" \
    --outpath "${OUT_DIR}" \
    --shards "${S0}:${S1}" \
    --algorithm clue \
    --clue-preset "${CLUE_PRESET}"
echo "done"
