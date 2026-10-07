#!/bin/bash
set -e
export PYTHONPATH=$(pwd)
export KERAS_BACKEND=torch

SMOKE_DIR=local_test_data/colliderml_smoke
rm -Rf -- "$SMOKE_DIR"
export TFDS_DATA_DIR=$(pwd)/"$SMOKE_DIR"/tensorflow_datasets

# Unit tests run in the once-per-run 'units' CI job (.github/workflows/test.yml), not here.

# 1. Synthetic fixture (offline)
uv run python3 -m mlpf.data.colliderml.make_test_fixture \
    --output "$SMOKE_DIR"/colliderml --num-events 20

# 2. Convert fixture -> MLPF parquet
uv run python3 -m mlpf.data.colliderml.postprocessing \
    --input "$SMOKE_DIR"/colliderml \
    --sample ttbar_pu0 \
    --outpath "$SMOKE_DIR"/colliderml_mlpf_parquet \
    --shards 0:2

# 3. Validate: unified key4hep-style validator (extended for colliderml)
uv run python3 tests/validate_parquet.py \
    --input "$SMOKE_DIR"/colliderml_mlpf_parquet/train-00000-of-00002.parquet \
    --detector colliderml \
    --max-events 20 --plots-dir plots --mode report

# 4. TFDS build (config 10; with 2 files, split_list puts them in the last split)
uv run tfds build mlpf/heptfds/colliderml_pf/ttbar --config 10 \
    --manual_dir "$SMOKE_DIR"/colliderml_mlpf_parquet --data_dir "$TFDS_DATA_DIR"

# 5. Training (CPU, 2 steps, reaches the plotting path via --pipeline for the colliderml branch)
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-colliderml-v1 \
  --production colliderml \
  --data-dir "$TFDS_DATA_DIR" \
  --prefix MLPF_colliderml_test_ \
  --pipeline \
  train \
  --num_steps 2 \
  --checkpoint_freq 1 \
  --gpus 0 \
  --dtype float32 \
  --gpu_batch_multiplier 1 \
  --ntrain 10 --ntest 10 --nvalid 10 \
  --num_workers 1 --prefetch_factor 1
