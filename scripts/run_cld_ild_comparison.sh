#!/usr/bin/env bash
set -euo pipefail

# Input layout: POST/{cld,ild}/{ttbar,qq,ww_fullhad}/*.parquet.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
POST=${1:?Usage: run_cld_ild_comparison.sh POST OUTPUT [EVENTS]}
OUTPUT=${2:?Usage: run_cld_ild_comparison.sh POST OUTPUT [EVENTS]}
EVENTS=${3:-200}
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/particleflow-matplotlib}"
NOTE="CLD and ILD FCC-ee at 365 GeV; $EVENTS events per process and detector. Both ROOT samples are converted with the current postprocessor. CLD uses its existing track truth associations; ILD uses fitted MarlinTrkTracks with direct tracker-hit majority truth (minimum purity 0.75). Generator truth jets precede detector visibility and target assignment; target jets contain retained training targets. Jet clustering is ee_genkt, R=0.4, p=-1, pT > 5 GeV for both detectors. Samples are compared statistically; shared seed labels do not establish identical generator events. Targets carry truth kinematics, so a unit target/truth jet response is not a detector energy-resolution measurement. ILD dEdx is unavailable and filled with zero. Jet spectra show counts per event; feature histograms show fractions per object."
ARGS=()
for sample in ttbar qq ww_fullhad; do
    for detector in cld ild; do
        ARGS+=(--parquet "$sample-$detector:$detector=$POST/$detector/$sample")
    done
done
.venv/bin/python scripts/compare_detector_features.py "${ARGS[@]}" \
    --num-events "$EVENTS" --process 'ttbar / qq / WW full-hadronic' \
    --title 'CLD versus ILD: three processes at 365 GeV' --note "$NOTE" --output-dir "$OUTPUT/all_samples"
for sample in ttbar qq ww_fullhad; do
    .venv/bin/python scripts/compare_detector_features.py \
        --parquet "cld=$POST/cld/$sample" --parquet "ild=$POST/ild/$sample" \
        --num-events "$EVENTS" --process "$sample" --title "CLD versus ILD: $sample at 365 GeV" \
        --note "$NOTE" --output-dir "$OUTPUT/$sample"
done
