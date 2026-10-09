#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-openlab}

OUTPUT_DIR=local_test_data/detector_comparison/event_displays
SOURCE_DIR=$OUTPUT_DIR/hires_sources
COLLIDERML_PARQUET=${COLLIDERML_DISPLAY_PARQUET:-local_test_data/detector_comparison/colliderml/train-00000-of-01000.parquet}
mkdir -p "$SOURCE_DIR"

.venv/bin/python scripts/visualize_key4hep.py \
    /mnt/work/mlpf/root/clic/ttbar/reco_p8_ee_ttbar_ecm380_300000.root \
    --events 0 1 --plot-limit 6500 --max-hits 800 --dpi 300 --output-dir "$SOURCE_DIR"
.venv/bin/python scripts/visualize_key4hep.py \
    /mnt/work/mlpf/root/cld/ttbar/reco_p8_ee_ttbar_ecm365_300000.root \
    --events 0 1 --plot-limit 6500 --max-hits 800 --dpi 300 --output-dir "$SOURCE_DIR"
.venv/bin/python scripts/visualize_key4hep.py \
    /mnt/work/mlpf/root/idea/ttbar/reco_p8_ee_ttbar_ecm365_300000.root \
    --events 0 1 --plot-limit 6500 --max-hits 800 --dpi 300 --output-dir "$SOURCE_DIR"
.venv/bin/python scripts/visualize_colliderml.py \
    --input local_test_data/detector_comparison/colliderml_source \
    --parquet "$COLLIDERML_PARQUET" \
    --events 0 1 --plot-limit 6500 --max-hits 800 --dpi 300 --output-dir "$SOURCE_DIR"

.venv/bin/python scripts/create_openlab_event_mosaic.py \
    --input-dir "$SOURCE_DIR" \
    --output "$OUTPUT_DIR/openlab_multi_event_dark_2x.png" \
    --events 0 1 --scale 2
