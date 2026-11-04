#!/bin/bash
#SBATCH --job-name=cld-granularity-scan
#SBATCH --partition=main
#SBATCH --account=hepusers
#SBATCH --time=48:00:00
#SBATCH --mem=2000
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/joosep/particleflow/reports/cld_granularity_scan/driver-%j.log

set -euo pipefail
cd /home/joosep/particleflow
export PATH=/home/joosep/.pixi/bin:$PATH
export MAX_JOBS_PER_POINT=20

scripts/cld_ecal_readout/launch_granularity_scan.sh all
