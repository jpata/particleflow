#!/usr/bin/env bash
# Convert local CI/real fixtures and validate the independent track/calo targets.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TRACK_CALO_PYTHON=${TRACK_CALO_PYTHON:-$PWD/.venv/bin/python}
TRACK_CALO_OUTPUT=${TRACK_CALO_OUTPUT:-$PWD/local_test_data/track_calo_targets}
NUM_EVENTS=${NUM_EVENTS:-3}
CLD_ROOT_FILE=${CLD_ROOT_FILE:-local_test_data/cld/p8_ee_ttbar_ecm365/root/reco_p8_ee_ttbar_ecm365_300000.root}
CLIC_ROOT_FILE=${CLIC_ROOT_FILE:-local_test_data/clic/p8_ee_ttbar_ecm380/root/reco_p8_ee_ttbar_ecm380_300000.root}
MAIA_ROOT_FILE=${MAIA_ROOT_FILE:-local_test_data/maia_smoke/root/ttbar_reco_10000.slcio.edm4hep.root}
IDEA_ROOT_FILE=${IDEA_ROOT_FILE:-/mnt/work/mlpf/root/idea/ttbar/reco_p8_ee_ttbar_ecm365_300000.root}
COLLIDERML_SOURCE=${COLLIDERML_SOURCE:-local_test_data/detector_comparison/colliderml_source}
mkdir -p "$TRACK_CALO_OUTPUT"
for detector in cld clic maia idea; do
    case "$detector" in
        cld) source_file=$CLD_ROOT_FILE ;;
        clic) source_file=$CLIC_ROOT_FILE ;;
        maia) source_file=$MAIA_ROOT_FILE ;;
        idea) source_file=$IDEA_ROOT_FILE ;;
    esac
    test -s "$source_file"
    echo "Converting $detector ($NUM_EVENTS events)"
    # Explicit local reader avoids the FSSpecSource PODIO metadata stall.
    "$TRACK_CALO_PYTHON" -u -c 'import functools, runpy, uproot; original_open = uproot.open; uproot.open = lambda *a, **kw: original_open(*a, **{"handler": uproot.source.file.MultithreadedFileSource, **kw}); runpy.run_module("mlpf.data.key4hep.postprocessing", run_name="__main__")' \
        --input "$source_file" --detector "$detector" --num-events "$NUM_EVENTS" \
        --outpath "$TRACK_CALO_OUTPUT/$detector" --track-calo-targets > "$TRACK_CALO_OUTPUT/$detector.log" 2>&1
done
echo "Converting ColliderML ($NUM_EVENTS events)"
"$TRACK_CALO_PYTHON" -m mlpf.data.colliderml.postprocessing --input "$COLLIDERML_SOURCE" \
    --sample ttbar_pu0 --outpath "$TRACK_CALO_OUTPUT/colliderml" --shards 0:1 \
    --num-events "$NUM_EVENTS" --track-calo-targets > "$TRACK_CALO_OUTPUT/colliderml.log" 2>&1
"$TRACK_CALO_PYTHON" scripts/studies/validate_track_calo_targets.py \
    "$TRACK_CALO_OUTPUT"/{cld,clic,maia,idea,colliderml}/*.parquet --output "$TRACK_CALO_OUTPUT/validation.json"
