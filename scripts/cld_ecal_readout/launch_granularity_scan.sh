#!/bin/bash
set -euo pipefail

group=${1:-all}
shift || true
max_jobs=${MAX_JOBS_PER_POINT:-50}

ecal=(cld_ecal_pitch_5p1 cld_ecal_pitch_4 cld_ecal_pitch_3 cld_ecal_pitch_2p5 cld_ecal_pitch_2 cld_ecal_pitch_1p5)
hcal=(cld_hcal_pitch_30 cld_hcal_pitch_25 cld_hcal_pitch_20 cld_hcal_pitch_15 cld_hcal_pitch_12 cld_hcal_pitch_10)

case "$group" in
    ecal) points=("${ecal[@]}") ;;
    hcal) points=("${hcal[@]}") ;;
    all) points=("${ecal[@]}" "${hcal[@]}") ;;
    *) echo "Usage: $0 {ecal|hcal|all} [extra Snakemake arguments]" >&2; exit 2 ;;
esac

for point in "${points[@]}"; do
    echo "Starting $point with at most $max_jobs active jobs" >&2
    PROD="$point" pixi run gen --jobs "$max_jobs" "$@"
done
