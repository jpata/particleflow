#!/bin/bash
#SBATCH --job-name=colliderml_download
#SBATCH --partition=genx
#SBATCH --time=78:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs_slurm/colliderml_download_%A_%a.out
#SBATCH --error=logs_slurm/colliderml_download_%A_%a.err
#SBATCH --export=NONE

set -euo pipefail

# Download ColliderML Release-1 configs (ttbar, pu0 or pu200) from Hugging Face to ceph.
# Adapted from collider-fm/slurm/download_colliderml.slurm.
#
# Usage:
#   # download all 4 object configs for pu0 (default), one array task per config:
#   sbatch --array=0-3%4 scripts/flatiron/colliderml_download.sh
#
#   # pu200 instead:
#   sbatch --array=0-3%4 scripts/flatiron/colliderml_download.sh pu200
#
#   # a single config, non-array:
#   sbatch scripts/flatiron/colliderml_download.sh pu200 particles
#
# No --max-events: full config via snapshot_download, which is parallel and
# resumable. Re-submitting the same task skips files already complete.
#
# Requires: the `colliderml` package (installed in this repo's uv environment) and an
# HF_TOKEN in .env in this repo (without it the Hub throttles to unauthenticated rate
# limits, which matters at this volume).

PU="${1:-pu0}"
OBJ="${2:-}"

OBJECTS=(calo_hits particles tracker_hits tracks)
if [ -n "${OBJ}" ]; then
  CONFIGS=("ttbar_${PU}_${OBJ}")
else
  CONFIGS=()
  for o in "${OBJECTS[@]}"; do
    CONFIGS+=("ttbar_${PU}_${o}")
  done
fi

# In array mode each task takes one config: task i -> CONFIGS[i].
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TASK_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
  IDX=$(( SLURM_ARRAY_TASK_ID - TASK_MIN ))
  if [ "${IDX}" -ge "${#CONFIGS[@]}" ]; then
    echo "array task ${SLURM_ARRAY_TASK_ID} is beyond the config list; nothing to do"
    exit 0
  fi
  CONFIG="${CONFIGS[${IDX}]}"
else
  if [ "${#CONFIGS[@]}" -ne 1 ]; then
    echo "ERROR: non-array mode needs a single object: $0 ${PU} <calo_hits|particles|tracker_hits|tracks>" >&2
    exit 2
  fi
  CONFIG="${CONFIGS[0]}"
fi

REVISION=64c3d2f112df3d5d20979d22da7cfdff13e10c4b

# The colliderml CLI lives in this repo's uv environment.
if [ ! -x .venv/bin/colliderml ]; then
  echo "ERROR: .venv/bin/colliderml not found in $(pwd)" >&2
  echo "  run 'uv sync' first (colliderml is a dependency of this repo)" >&2
  exit 2
fi
source .venv/bin/activate

# Load tokens (e.g. HF_TOKEN) from the repo .env if present, else from the environment.
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN not set; downloads will be rate-limited" >&2
fi

# Download needs network access; force online regardless of the submitting
# shell's environment (sbatch --export=ALL can carry over HF_HUB_OFFLINE=1
# from a prior session).
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# Keep both the payload and the hf_xet chunk cache off the home filesystem.
export COLLIDERML_DATA_DIR=/mnt/ceph/users/ewulff/data/colliderml
export HF_HOME=/mnt/ceph/users/ewulff/huggingface

echo "Downloading ${CONFIG} @ ${REVISION} -> ${COLLIDERML_DATA_DIR}"

colliderml download \
    --config "${CONFIG}" \
    --revision "${REVISION}" \
    --out "${COLLIDERML_DATA_DIR}"

# Sanity check: every Release-1 config is 1000 shards (1000 events/shard for pu0, 100 for
# pu200 — the probe below prints the actual count).
SHARD_DIR="${COLLIDERML_DATA_DIR}/CERN__ColliderML-Release-1/${CONFIG}/data/${CONFIG}"
N_SHARDS=$(ls "${SHARD_DIR}"/train-*.parquet 2>/dev/null | wc -l)
echo "${CONFIG}: ${N_SHARDS}/1000 shards in ${SHARD_DIR}"
if [ "${N_SHARDS}" -ne 1000 ]; then
    echo "ERROR: expected 1000 shards for ${CONFIG}, found ${N_SHARDS}" >&2
    exit 1
fi

python3 -c "
import pyarrow.parquet as pq
f = pq.ParquetFile('${SHARD_DIR}/train-00000-of-01000.parquet')
print('shard 0:', f.metadata.num_rows, 'events,', len(f.schema_arrow), 'columns')
"
echo "done"
