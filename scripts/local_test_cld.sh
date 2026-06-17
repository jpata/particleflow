#!/bin/bash
set -e
export TFDS_DATA_DIR=$(pwd)/tensorflow_datasets
export PYTHONPATH=$(pwd)
export KERAS_BACKEND=torch

# Quick unit tests
uv run python -m pytest --cache-clear tests

# 1. Create data directories
rm -Rf local_test_data/cld
mkdir -p local_test_data/cld/p8_ee_ttbar_ecm365/root
cd local_test_data/cld/p8_ee_ttbar_ecm365/root

# 2. Download CLD ROOT files
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cld/v1.2.3_key4hep_2025-05-29_CLD_f1e8f9/gen/root/reco_p8_ee_ttbar_ecm365_300000.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cld/v1.2.3_key4hep_2025-05-29_CLD_f1e8f9/gen/root/reco_p8_ee_ttbar_ecm365_300001.root

cd ../../../..

# 3. Postprocessing
for file in local_test_data/cld/p8_ee_ttbar_ecm365/root/*.root; do
  uv run python3 mlpf/data/key4hep/postprocessing.py \
    --input $file \
    --outpath local_test_data/cld/p8_ee_ttbar_ecm365 \
    --detector cld
done

# Run Postprocessing validation
# Find the first parquet file for validation
SAMPLE_PARQUET=$(ls local_test_data/cld/p8_ee_ttbar_ecm365/*.parquet | head -n 1)
uv run python3 tests/visualize_pn.py $SAMPLE_PARQUET 0
uv run python3 tests/validate_inclusive_hits.py $SAMPLE_PARQUET --bfield 2.0

# 4. TFDS Build
# Using config 10 because with only 2 files, split_list puts them in the last (10th) split
uv run tfds build mlpf/heptfds/cld_pf_edm4hep/ttbar --config 10 --manual_dir ./local_test_data/cld --data_dir ./tensorflow_datasets

# 5. Training
uv run python3 mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cld-v1 \
  --production cld \
  --data-dir ./tensorflow_datasets/ \
  --prefix MLPF_cld_test_ \
  --pipeline \
  train \
  --num_steps 2 \
  --checkpoint_freq 1 \
  --gpus 0 \
  --dtype float32 \
  --ntrain 10 --ntest 10 --nvalid 10 \
  --num_workers 1 --prefetch_factor 1
