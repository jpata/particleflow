#!/bin/bash
set -e

# This script downloads a small amount of ROOT data and runs postprocessing
# to generate the .parquet files needed by the validation scripts.
# Based on scripts/local_test_cld.sh.
# Pin the Hugging Face revision that contains both files required by the TFDS train and test splits.
fixture_url="https://huggingface.co/datasets/jpata/particleflow/resolve/04a1da49a663e3704e2af4a3d1d0e26ad9369c30/root/cld/ttbar"

export PYTHONPATH=$(pwd)

# CLD Data (Parquet)
echo "Setting up CLD test data..."
mkdir -p local_test_data/cld/p8_ee_ttbar_ecm365/root
pushd local_test_data/cld/p8_ee_ttbar_ecm365/root > /dev/null
wget -nv -nc "$fixture_url/reco_p8_ee_ttbar_ecm365_300000.root"
wget -nv -nc "$fixture_url/reco_p8_ee_ttbar_ecm365_300001.root"
popd > /dev/null

for file in local_test_data/cld/p8_ee_ttbar_ecm365/root/*.root; do
  uv run python3 mlpf/data/key4hep/postprocessing.py \
    --input $file \
    --outpath local_test_data/cld/p8_ee_ttbar_ecm365 \
    --detector cld
done

echo "CLD test data setup complete."
echo "You can now run validation scripts, for example:"
echo "  uv run python3 tests/validate_parquet.py --input local_test_data/cld/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_300000.parquet --detector cld"
