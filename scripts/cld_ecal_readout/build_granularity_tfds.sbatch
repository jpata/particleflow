#!/bin/bash
#SBATCH --job-name=cld-granularity-tfds
#SBATCH --partition=main
#SBATCH --account=hepusers
#SBATCH --array=0-11%2
#SBATCH --time=48:00:00
#SBATCH --mem=2000
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/joosep/particleflow/reports/cld_granularity_scan/tfds/controller-%A_%a.log

set -euo pipefail
cd /home/joosep/particleflow
export PATH=/home/joosep/.pixi/bin:$PATH

points=(cld_ecal_pitch_5p1 cld_ecal_pitch_4 cld_ecal_pitch_3 cld_ecal_pitch_2p5 cld_ecal_pitch_2 cld_ecal_pitch_1p5
        cld_hcal_pitch_30 cld_hcal_pitch_25 cld_hcal_pitch_20 cld_hcal_pitch_15 cld_hcal_pitch_12 cld_hcal_pitch_10)
point=${points[$SLURM_ARRAY_TASK_ID]}

# The production's dataset_only setting generates only post/TFDS rules. Each
# stage completes before the next starts; rerunning skips finished outputs.
PROD=$point pixi run post --jobs 20
PROD=$point pixi run tfds --jobs 5
PROD=$point pixi run tfds_hit --jobs 3
