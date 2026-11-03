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

# Download ColliderML Release-1 configs from Hugging Face to ceph.
# Adapted from collider-fm/slurm/download_colliderml.slurm.
#
# Usage:
#   # download all 4 object configs of ttbar_pu0 (default), one array task per config:
#   sbatch --array=0-3%1 scripts/flatiron/colliderml_download.sh
#
#   # ttbar_pu200 instead (a bare pileup label means ttbar):
#   sbatch --array=0-3%1 scripts/flatiron/colliderml_download.sh pu200
#
#   # any other Release-1 sample, named <process>_<pileup>:
#   sbatch --array=0-3%1 scripts/flatiron/colliderml_download.sh zee_pu200
#
#   # a single config, non-array:
#   sbatch scripts/flatiron/colliderml_download.sh pu200 particles
#
# Release-1 samples at the pinned revision: ttbar_{pu0,pu200}, dihiggs_{pu0,pu200},
# ggf_{pu0,pu200}, hidden_valley_pu0, higgs_portal_pu0, pileup_only_pu0, zprime_pu0,
# zee_pu200, zmumu_pu200. All configs have 1000 shards except dihiggs_pu200_tracker_hits (295).
# scripts/flatiron/colliderml_process_samples.sh chains download -> convert -> TFDS for them.
#
# No --max-events: full config via snapshot_download, which is parallel and
# resumable. Re-submitting the same task skips files already complete.
#
# Requires: the `colliderml` package (installed in this repo's uv environment) and an
# HF_TOKEN in .env in this repo (without it the Hub throttles to unauthenticated rate
# limits, which matters at this volume).

SAMPLE="${1:-pu0}"
OBJ="${2:-}"
# a bare pileup label is the ttbar sample (the original usage)
case "${SAMPLE}" in
  pu0|pu200) SAMPLE="ttbar_${SAMPLE}" ;;
esac

OBJECTS=(calo_hits particles tracker_hits tracks)
if [ -n "${OBJ}" ]; then
  CONFIGS=("${SAMPLE}_${OBJ}")
else
  CONFIGS=()
  for o in "${OBJECTS[@]}"; do
    CONFIGS+=("${SAMPLE}_${o}")
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
    echo "ERROR: non-array mode needs a single object: $0 ${SAMPLE} <calo_hits|particles|tracker_hits|tracks>" >&2
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

# Bandwidth: hf_xet adaptively raises its parallel range fetches until the link saturates; SCC
# asked us (2026-10-30) to stay within 1-2 Gbps after 12 tasks reached 5-8 Gbps on one node.
# Pin the per-process concurrency low and run few tasks at once (colliderml_process_samples.sh
# defaults to one task in total). One task measured (wchar) ~1.9 Gbps at concurrency 4 and
# ~1.5 Gbps at 3, hence the default 3. .env is read at job start, so HF_XET_FIXED_DOWNLOAD_CONCURRENCY=<n> there
# retunes already-submitted, not-yet-started tasks.
export HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false
export HF_XET_FIXED_DOWNLOAD_CONCURRENCY="${HF_XET_FIXED_DOWNLOAD_CONCURRENCY:-3}"

SHARD_DIR="${COLLIDERML_DATA_DIR}/CERN__ColliderML-Release-1/${CONFIG}/data/${CONFIG}"

# One download attempt plus the completeness check: the local shard count must equal the pinned
# revision's (1000 for every Release-1 config except dihiggs_pu200_tracker_hits, 295). Called
# in an `until` condition, where set -e is off, hence the explicit `|| return 1`.
download_complete() {
  colliderml download \
      --config "${CONFIG}" \
      --revision "${REVISION}" \
      --out "${COLLIDERML_DATA_DIR}" || return 1
  N_SHARDS=$(ls "${SHARD_DIR}"/train-*.parquet 2>/dev/null | wc -l)
  # list only data/<config> (1-2 API calls), not the whole ~48k-file repo
  N_EXPECTED=$(python3 -c "
from huggingface_hub import HfApi
entries = HfApi().list_repo_tree('CERN/ColliderML-Release-1', path_in_repo='data/${CONFIG}', repo_type='dataset', revision='${REVISION}')
print(sum(e.path.endswith('.parquet') for e in entries))
") || return 1
  echo "${CONFIG}: ${N_SHARDS}/${N_EXPECTED} shards in ${SHARD_DIR}"
  [ "${N_SHARDS}" -gt 0 ] && [ "${N_SHARDS}" -eq "${N_EXPECTED}" ]
}

# Each file fetch requests a xet read token from the Hub API, so parallel downloads exceed the
# per-token API rate limit (HTTP 429) regularly. The download resumes (complete files are
# skipped), so retry after the 5-minute rate-limit window for as long as attempts make progress;
# give up only after MAX_STALLED consecutive attempts that added no shard.
MAX_STALLED=6
STALLED=0
ATTEMPT=1
LAST_N=-1
echo "Downloading ${CONFIG} @ ${REVISION} -> ${COLLIDERML_DATA_DIR}"
until download_complete; do
  N_NOW=$(ls "${SHARD_DIR}"/train-*.parquet 2>/dev/null | wc -l || true)
  if [ "${N_NOW}" -gt "${LAST_N}" ]; then STALLED=0; else STALLED=$(( STALLED + 1 )); fi
  LAST_N="${N_NOW}"
  if [ "${STALLED}" -ge "${MAX_STALLED}" ]; then
    echo "ERROR: ${CONFIG} stuck at ${N_NOW} shards for ${STALLED} attempts; resubmit to resume" >&2
    exit 1
  fi
  echo "attempt ${ATTEMPT} failed or incomplete (${N_NOW} shards so far, e.g. HTTP 429); retrying in 5 min" >&2
  ATTEMPT=$(( ATTEMPT + 1 ))
  sleep 300
done

# Events per shard: 1000 for pu0 (10000 for pileup_only_pu0), 100 for pu200.
python3 -c "
import pyarrow.parquet as pq
from pathlib import Path
f = pq.ParquetFile(sorted(Path('${SHARD_DIR}').glob('train-*.parquet'))[0])
print('first shard:', f.metadata.num_rows, 'events,', len(f.schema_arrow), 'columns')
"
echo "done"
