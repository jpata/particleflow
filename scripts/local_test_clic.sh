#!/bin/bash
set -e
export TFDS_DATA_DIR=$(pwd)/tensorflow_datasets
export PYTHONPATH=$(pwd)
export KERAS_BACKEND=torch

# Quick unit tests
uv run python -m pytest --cache-clear tests

# 1. Fetch test data
rm -Rf local_test_data/clic
./scripts/fetch_test_data_clic.sh

# Run Postprocessing validation
# Find the first parquet file for validation
SAMPLE_PARQUET=$(ls local_test_data/clic/p8_ee_ttbar_ecm380/*.parquet | head -n 1)
uv run python3 tests/visualize_pn.py $SAMPLE_PARQUET 0
uv run python3 tests/validate_inclusive_hits.py $SAMPLE_PARQUET --bfield 4.0

# Collect validation plots for the CI artifact upload
mkdir -p plots
cp pn_validation_side_*.png plots/
cp unified_validation.png plots/unified_validation_clic.png

# 4. TFDS Build
# Using config 10 because with only 2 files, split_list puts them in the last (10th) split
uv run tfds build mlpf/heptfds/clic_pf_edm4hep/ttbar --config 10 --manual_dir ./local_test_data/clic --data_dir ./tensorflow_datasets

# 5. Training
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-clic-v1 \
  --production clic \
  --data-dir ./tensorflow_datasets/ \
  --prefix MLPF_clic_test_ \
  --pipeline \
  train \
  --num_steps 2 \
  --checkpoint_freq 1 \
  --gpus 0 \
  --dtype float32 \
  --ntrain 10 --ntest 10 --nvalid 10 \
  --num_workers 1 --prefetch_factor 1
