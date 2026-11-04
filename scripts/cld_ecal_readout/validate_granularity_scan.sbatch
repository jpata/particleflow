#!/bin/bash
#SBATCH --job-name=cld-granularity-validate
#SBATCH --partition=main
#SBATCH --account=hepusers
#SBATCH --array=0-11%6
#SBATCH --time=02:00:00
#SBATCH --mem=4000
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/joosep/particleflow/reports/cld_granularity_scan/validation/validate-%A_%a.log

set -euo pipefail
cd /home/joosep/particleflow
export PATH=/home/joosep/.local/bin:$PATH
export MPLCONFIGDIR=/tmp/cld-mpl-validate-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export UV_CACHE_DIR=/tmp/uv-cache-cld-validate-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
points=(ecal_5p1 ecal_4 ecal_3 ecal_2p5 ecal_2 ecal_1p5 hcal_30 hcal_25 hcal_20 hcal_15 hcal_12 hcal_10)
uv run --no-sync python scripts/cld_ecal_readout/validate_granularity_scan.py --point "${points[$SLURM_ARRAY_TASK_ID]}"
