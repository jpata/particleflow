#!/bin/bash
set -e

# This script downloads a small amount of ROOT data and runs postprocessing
# to generate the .pkl files needed by the validation scripts.
# Based on scripts/local_test_cms.sh.

export PYTHONPATH=$(pwd)

# CMS Data (Pickle)
echo "Setting up CMS test data..."
mkdir -p local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/root
pushd local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/root > /dev/null
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/20240823_simcluster/pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_100000.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/20240823_simcluster/pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_100001.root
popd > /dev/null

for file in local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/root/*.root; do
  uv run python3 mlpf/data/cms/postprocessing2.py \
    --input $file \
    --outpath local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi
done

echo "CMS test data setup complete."
echo "You can now run validation scripts, for example:"
echo "  uv run python3 tests/validate_cms_gt.py --input local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/pfntuple_100000.pkl"
