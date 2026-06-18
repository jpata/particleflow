#!/bin/bash
set -e

# This script downloads a small amount of ROOT data and runs postprocessing
# to generate the .parquet files needed by the validation scripts.
# Based on scripts/local_test_cld.sh.

export PYTHONPATH=$(pwd)

# CLD Data (Parquet)
echo "Setting up CLD test data..."
mkdir -p local_test_data/cld/p8_ee_ttbar_ecm365/root
pushd local_test_data/cld/p8_ee_ttbar_ecm365/root > /dev/null
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cld/v1.2.3_key4hep_2025-05-29_CLD_f1e8f9/gen/root/reco_p8_ee_ttbar_ecm365_300000.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cld/v1.2.3_key4hep_2025-05-29_CLD_f1e8f9/gen/root/reco_p8_ee_ttbar_ecm365_300001.root
popd > /dev/null

for file in local_test_data/cld/p8_ee_ttbar_ecm365/root/*.root; do
  uv run python3 mlpf/data/key4hep/postprocessing.py \
    --input $file \
    --outpath local_test_data/cld/p8_ee_ttbar_ecm365 \
    --detector cld
done

echo "CLD test data setup complete."
echo "You can now run validation scripts, for example:"
echo "  uv run python3 tests/validate_parquet_gt.py --input local_test_data/cld/p8_ee_ttbar_ecm365/reco_p8_ee_ttbar_ecm365_300000.parquet"
