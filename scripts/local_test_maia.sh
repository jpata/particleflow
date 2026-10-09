#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$PWD"
export KERAS_BACKEND=torch
SMOKE_DIR=local_test_data/maia_smoke
NUM_EVENTS=${MAIA_NUM_EVENTS:-100}
ROOT_FILE=${MAIA_ROOT_FILE:-$SMOKE_DIR/root/ttbar_reco_10000.slcio.edm4hep.root}
ROOT_BASENAME=$(basename "$ROOT_FILE")
# The converter keeps the portion of the basename before the first dot.
PARQUET_FILE=$SMOKE_DIR/parquet/${ROOT_BASENAME%%.*}.parquet
export TFDS_DATA_DIR="$PWD/$SMOKE_DIR/tensorflow_datasets"
mkdir -p "$SMOKE_DIR/root" "$SMOKE_DIR/parquet"

# The public sample host uses a self-signed certificate. No credentials are sent.
if [ ! -s "$ROOT_FILE" ]; then
    curl --insecure --fail --location --retry 2 --retry-all-errors --continue-at - \
        --speed-limit 100000 --speed-time 30 --max-time 300 \
        --output "$ROOT_FILE.part" \
        'https://uaf-3.t2.ucsd.edu/~atuna/muoncollider/data/mlpf/ttbar/v04/ttbar_reco_10000.slcio.edm4hep.root' || \
    curl --fail --location --retry 5 --retry-all-errors --continue-at - \
        --output "$ROOT_FILE.part" \
        'http://uaf-3.t2.ucsd.edu/~atuna/muoncollider/data/mlpf/ttbar/v04/ttbar_reco_10000.slcio.edm4hep.root'
    mv -- "$ROOT_FILE.part" "$ROOT_FILE"
fi

# Respect small supplied samples as well as the requested smoke-test limit.
NUM_EVENTS=$(uv run python3 -c 'import sys, uproot; f = uproot.open(sys.argv[1], handler=uproot.source.file.MultithreadedFileSource); print(min(int(sys.argv[2]), f["events"].num_entries))' "$ROOT_FILE" "$NUM_EVENTS")
echo "MAIA smoke test: $NUM_EVENTS events from $ROOT_FILE"

# Select Uproot's local reader to avoid the FSSpecSource PODIO metadata stall.
PARQUET_EVENTS=0
if [ -s "$PARQUET_FILE" ]; then
    PARQUET_EVENTS=$(uv run python3 -c 'import sys, awkward as ak; print(len(ak.from_parquet(sys.argv[1], columns=["X_track"])["X_track"]))' "$PARQUET_FILE")
fi
if [ "$PARQUET_EVENTS" != "$NUM_EVENTS" ] || [ "${MAIA_RECONVERT:-0}" = 1 ]; then
    uv run python3 -u -c 'import functools, runpy, uproot; uproot.open = functools.partial(uproot.open, handler=uproot.source.file.MultithreadedFileSource); runpy.run_module("mlpf.data.key4hep.postprocessing", run_name="__main__")' \
        --input "$ROOT_FILE" --outpath "$SMOKE_DIR/parquet" \
        --detector maia --num-events "$NUM_EVENTS"
fi

uv run python3 tests/validate_parquet.py \
    --input "$PARQUET_FILE" --detector maia --max-events "$NUM_EVENTS" \
    --plots-dir "$SMOKE_DIR/validation_plots" --mode report

# One source file is split into two event-disjoint parquet shards before TFDS.
uv run python3 scripts/build_colliderml_comparison_tfds.py \
    --detector maia --input "$PARQUET_FILE" \
    --manual-dir "$SMOKE_DIR/tfds_manual" --data-dir "$TFDS_DATA_DIR"

uv run python3 mlpf/pipeline.py \
    --spec-file particleflow_spec.yaml --model-name pyg-maia-v1 \
    --production-name maia --data-dir "$TFDS_DATA_DIR" \
    --experiments-dir "$SMOKE_DIR/experiments" --prefix MLPF_maia_test_ --pipeline \
    train --num_steps 2 --checkpoint_freq 1 --gpus 0 --dtype float32 \
    --gpu_batch_multiplier 1 --ntrain 10 --ntest 10 --nvalid 10 \
    --num_workers 1 --prefetch_factor 1
