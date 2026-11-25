#!/usr/bin/env bash
set -euo pipefail

# Outputs are kept separate from production data.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$PWD"
export KERAS_BACKEND=torch
WORK=local_test_data/detector_comparison
# Key4hep inputs are site-local: the ROOT files (<root>/<detector>/ttbar/) and stored TFDS
# version directories. Override with environment variables on another machine.
KEY4HEP_ROOT_DIR=${KEY4HEP_ROOT_DIR:-/mnt/work/mlpf/root}
CLIC_TFDS=${CLIC_TFDS:-/mnt/work/mlpf/tensorflow_datasets/clic/clic_edm_ttbar_pf/1/3.3.0}
CLD_TFDS=${CLD_TFDS:-/mnt/work/mlpf/tensorflow_datasets/cld/cld_edm_ttbar_pf/1/3.3.0}
IDEA_TFDS=${IDEA_TFDS:-/mnt/work/mlpf/tensorflow_datasets/idea/idea_edm_ttbar_pf/1/0.3.0}
# ColliderML and MAIA are built here at their builders' current versions.
builder_version() { sed -n 's/.*VERSION = tfds.core.Version(.*"\([0-9.]*\)").*/\1/p' "$1"; }
COLLIDERML_VERSION=$(builder_version mlpf/heptfds/colliderml_pf/ttbar.py)
MAIA_VERSION=$(builder_version mlpf/heptfds/maia_pf_edm4hep/ttbar.py)
# Isolate builds when converter/feature/TFDS logic changes. Old data is preserved.
COLLIDERML_FINGERPRINT=$(sha256sum mlpf/data/colliderml/{postprocessing,tracks,reader,truth,clustering}.py \
    mlpf/data/target_building.py mlpf/heptfds/colliderml_pf/ttbar.py \
    mlpf/heptfds/colliderml_utils/utils.py mlpf/conf.py | sha256sum | cut -c1-12)
COLLIDERML_BUILD="$WORK/colliderml_builds/$COLLIDERML_FINGERPRINT"
mkdir -p "$WORK"/{clic,cld,idea}

if [ ! -f "$WORK/colliderml_source/provenance.json" ]; then
    HF_HOME="$PWD/$WORK/huggingface_cache" .venv/bin/python scripts/download_colliderml_comparison_sample.py \
        --output-dir "$WORK/colliderml_source"
fi

for detector in clic cld idea; do
    energy=365
    if [ "$detector" = clic ]; then energy=380; fi
    filename="reco_p8_ee_ttbar_ecm${energy}_300000"
    if [ ! -f "$WORK/$detector/$filename.parquet" ]; then
        # Uproot 5.7.2's default FSSpecSource stalls on these files' PODIO string
        # metadata. Its local file reader succeeds; the physics conversion is unchanged.
        .venv/bin/python -u -c 'import functools, runpy, uproot; uproot.open = functools.partial(uproot.open, handler=uproot.source.file.MultithreadedFileSource); runpy.run_module("mlpf.data.key4hep.postprocessing", run_name="__main__")' \
            --input "$KEY4HEP_ROOT_DIR/$detector/ttbar/$filename.root" \
            --outpath "$WORK/$detector" --detector "$detector" --num-events 100
    fi
done

if [ ! -f "$COLLIDERML_BUILD/parquet/train-00000-of-01000.parquet" ]; then
    .venv/bin/python -m mlpf.data.colliderml.postprocessing \
        --input "$WORK/colliderml_source" --sample ttbar_pu0 \
        --outpath "$COLLIDERML_BUILD/parquet" --shards 0:1 --num-events 100
fi

.venv/bin/python scripts/build_colliderml_comparison_tfds.py \
    --input "$COLLIDERML_BUILD/parquet/train-00000-of-01000.parquet" \
    --manual-dir "$COLLIDERML_BUILD/manual" --data-dir "$COLLIDERML_BUILD/tensorflow_datasets"

MAIA_TFDS=local_test_data/maia_smoke/tensorflow_datasets/maia_edm_ttbar_pf/10/$MAIA_VERSION
if [ ! -f "$MAIA_TFDS/dataset_info.json" ]; then
    bash scripts/local_test_maia.sh
fi

COLLIDERML_DISPLAY_PARQUET="$COLLIDERML_BUILD/parquet/train-00000-of-01000.parquet" \
    bash scripts/run_detector_event_displays.sh

# Stored Key4hep TFDS columns are optional: skip any that are not available on this machine.
TFDS_ARGS=(--tfds "colliderml=$COLLIDERML_BUILD/tensorflow_datasets/colliderml_ttbar_nopu_pf/10/$COLLIDERML_VERSION")
for spec in "clic=$CLIC_TFDS" "cld=$CLD_TFDS" "idea=$IDEA_TFDS"; do
    if [ -f "${spec#*=}/dataset_info.json" ]; then
        TFDS_ARGS+=(--tfds "$spec")
    else
        echo "Skipping ${spec%%=*} TFDS: ${spec#*=} not found" >&2
    fi
done
TFDS_ARGS+=(--tfds "maia=$MAIA_TFDS")

.venv/bin/python scripts/compare_detector_features.py \
    --parquet "colliderml=$COLLIDERML_BUILD/parquet" \
    --parquet "clic=$WORK/clic" --parquet "cld=$WORK/cld" --parquet "idea=$WORK/idea" \
    --parquet maia=local_test_data/maia_smoke/parquet \
    "${TFDS_ARGS[@]}" \
    --num-events 100 --split train+test --verify-colliderml-roundtrip --verify-maia-roundtrip \
    --maia-validation-report local_test_data/maia_smoke/validation_plots/validation_report.json \
    --output-dir "$WORK/real_plots"
