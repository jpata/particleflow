#!/bin/bash
#SBATCH --job-name=colliderml_convert
#SBATCH --partition=gen
#SBATCH -C icelake
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs_slurm/colliderml_convert_%j.out
#SBATCH --error=logs_slurm/colliderml_convert_%j.err
#SBATCH --export=NONE

# Convert a shard range on one node, one rank per shard (ntasks split the range evenly).
# Any rank failing nixes the whole submission; resume-with-skip keeps already-written
# shards, so a rerun picks up only the lost ones.
#
# Usage:
#   sbatch scripts/flatiron/colliderml_convert.sh                # full 1000 shards, ttbar_pu0
#   sbatch scripts/flatiron/colliderml_convert.sh 0 1000 pu200   # full range, ttbar_pu200
#   sbatch --ntasks=64 scripts/flatiron/colliderml_convert.sh 0 64  # single-node test
#   sbatch scripts/flatiron/colliderml_convert.sh 0 64 pu0 /path/outdir  # custom output dir
#
# Third arg is the pileup label (default pu0); the sample is ttbar_${PU} and output goes to
# mlpf_parquet/clustered/ttbar_${PU}/ unless a 4th arg overrides it.
# The per-rank shard arithmetic uses SLURM_NTASKS, so override --ntasks to taste.
# pu200 events are memory-heavy (~10-20 GB/rank); if the node OOMs, give ranks a larger
# memory slice, e.g. --ntasks=32 --cpus-per-task=2.
#

set -euo pipefail

S0="${1:-0}"
S1="${2:-1000}"
PU="${3:-pu0}"
# Optional 4th positional arg: output dir override. Note the job runs with --export=NONE, so
# COLLIDERML_OUT_DIR below never survives sbatch — it only helps direct/local runs.
OUTDIR_ARG="${4:-}"

if [ -z "${SLURM_NTASKS:-}" ]; then
  echo "expected to run under sbatch --ntasks" >&2
  exit 2
fi
RANGE=$(( S1 - S0 ))
N="$SLURM_NTASKS"
PER_RANK=$(( RANGE / N ))
EXTRA=$(( RANGE % N ))
if [ "$PER_RANK" -eq 0 ]; then
  echo "shard range [${S0}, ${S1}) has ${RANGE} shards, fewer than n_tasks=${N}; lower --ntasks" >&2
  exit 2
fi
echo "host $(hostname) sample=ttbar_${PU} shards [${S0}, ${S1}); ranks=${N}; ${EXTRA} rank(s) get ${PER_RANK}+1 shards, the rest ${PER_RANK}"

# allocation summary, printed before srun so it lands in the job log
for ((r=0; r<N; r++)); do
  RSTART=$(( S0 + r*PER_RANK + (r < EXTRA ? r : EXTRA) ))
  RLEN=$(( PER_RANK + (r < EXTRA ? 1 : 0) ))
  echo "  rank ${r}: shards [${RSTART}, $(( RSTART + RLEN )))"
done

CONDA_ENV=/mnt/home/ewulff/miniforge3/envs/mlpf
PYTHON=${CONDA_ENV}/bin/python
export PYTHONPATH="/mnt/home/ewulff/repositories/particleflow"
export KERAS_BACKEND=torch

SAMPLE="ttbar_${PU}"
SOURCE_DIR="/mnt/ceph/users/ewulff/data/colliderml/CERN__ColliderML-Release-1"
# COLLIDERML_OUT_DIR overrides the output root (default: the production clustered dir);
# the optional 4th positional arg wins over it.
OUT_DIR="${OUTDIR_ARG:-${COLLIDERML_OUT_DIR:-/mnt/ceph/users/ewulff/data/colliderml/mlpf_parquet/clustered/${SAMPLE}}}"
mkdir -p "${OUT_DIR}"
LOGDIR="/mnt/home/ewulff/repositories/particleflow/logs_slurm"
mkdir -p "$LOGDIR"

# rank r starts after r*PER_RANK shards plus the extra shards handed to ranks < r
# --export=NONE leaves PATH unset in the job environment, so srun cannot resolve `bash`;
# use the absolute path (the batch script itself runs via its shebang and is unaffected).
srun --kill-on-bad-exit=1 /bin/bash -c "
  RANK=\$SLURM_PROCID
  START=\$(( $S0 + RANK * $PER_RANK + (RANK < $EXTRA ? RANK : $EXTRA) ))
  LEN=\$(( $PER_RANK + (RANK < $EXTRA ? 1 : 0) ))
  END=\$(( START + LEN ))
  $PYTHON -m mlpf.data.colliderml.postprocessing \
    --input '$SOURCE_DIR' \
    --sample '$SAMPLE' \
    --outpath '$OUT_DIR' \
    --shards \"\$START:\$END\" \
    --algorithm bfs_merge \
    --merge-frac 0.25 \
    > \"${LOGDIR}/colliderml_convert.${SLURM_JOB_ID}.rank\${RANK}.out\"
"
echo "done"
