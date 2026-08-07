#!/bin/bash
set -e

# This script downloads a small amount of ROOT data and runs postprocessing
# to generate the .parquet files needed by the validation scripts.
# Based on scripts/local_test_clic.sh.
# ROOT files are hosted on HuggingFace:
# https://huggingface.co/datasets/jpata/particleflow/tree/main/root/clic/ttbar

export PYTHONPATH=$(pwd)

# CLIC Data (Parquet)
echo "Setting up CLIC test data..."
mkdir -p local_test_data/clic/p8_ee_ttbar_ecm380/root
pushd local_test_data/clic/p8_ee_ttbar_ecm380/root > /dev/null
wget -q --no-check-certificate -nc https://huggingface.co/datasets/jpata/particleflow/resolve/main/root/clic/ttbar/reco_p8_ee_ttbar_ecm380_300000.root
wget -q --no-check-certificate -nc https://huggingface.co/datasets/jpata/particleflow/resolve/main/root/clic/ttbar/reco_p8_ee_ttbar_ecm380_300001.root
popd > /dev/null

for file in local_test_data/clic/p8_ee_ttbar_ecm380/root/*.root; do
  uv run python3 mlpf/data/key4hep/postprocessing.py \
    --input $file \
    --outpath local_test_data/clic/p8_ee_ttbar_ecm380 \
    --detector clic
done

echo "CLIC test data setup complete."
echo "You can now run validation scripts, for example:"
echo "  uv run python3 tests/validate_inclusive_hits.py local_test_data/clic/p8_ee_ttbar_ecm380/reco_p8_ee_ttbar_ecm380_300000.parquet --bfield 4.0"
