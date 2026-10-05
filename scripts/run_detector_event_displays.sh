#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
OUTPUT_DIR=${EVENT_DISPLAY_DIR:-local_test_data/detector_comparison/event_displays}
COLLIDERML_PARQUET=${COLLIDERML_DISPLAY_PARQUET:-local_test_data/detector_comparison/colliderml/train-00000-of-01000.parquet}
events=(0 1 2)
if [ "$#" -gt 0 ]; then events=("$@"); fi

.venv/bin/python scripts/visualize_key4hep.py \
    /mnt/work/mlpf/root/clic/ttbar/reco_p8_ee_ttbar_ecm380_300000.root \
    /mnt/work/mlpf/root/cld/ttbar/reco_p8_ee_ttbar_ecm365_300000.root \
    /mnt/work/mlpf/root/idea/ttbar/reco_p8_ee_ttbar_ecm365_300000.root \
    local_test_data/maia_smoke/root/ttbar_reco_10000.slcio.edm4hep.root \
    --events "${events[@]}" --plot-limit 6500 --max-hits 800 --output-dir "$OUTPUT_DIR"

.venv/bin/python scripts/visualize_colliderml.py \
    --input local_test_data/detector_comparison/colliderml_source \
    --parquet "$COLLIDERML_PARQUET" \
    --events "${events[@]}" --plot-limit 6500 --max-hits 800 --output-dir "$OUTPUT_DIR"

.venv/bin/python scripts/build_detector_event_gallery.py --input "$OUTPUT_DIR" --events "${events[@]}"
