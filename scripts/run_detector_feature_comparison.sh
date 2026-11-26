#!/usr/bin/env bash
set -euo pipefail

# Outputs are kept separate from production data.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$PWD"
export KERAS_BACKEND=torch
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/particleflow-matplotlib}"

# Usage:
#   run_detector_feature_comparison.sh
#       ttbar report for ColliderML, CLIC, CLD, IDEA, ILD and MAIA, converted from ROOT.
#   run_detector_feature_comparison.sh --samples POST OUTPUT [EVENTS] [DETECTORS]
#       Key4hep detectors (default "cld ild") on each physics sample already converted into
#       POST/<detector>/<sample>/; one combined report and one report per sample.
if [ "${1:-}" = "--samples" ]; then
    usage="Usage: $0 --samples POST OUTPUT [EVENTS] [DETECTORS]"
    POST=${2:?$usage}
    OUTPUT=${3:?$usage}
    EVENTS=${4:-200}
    read -r -a detectors <<<"${5:-cld ild}"
    samples=()
    for sample in ttbar ww_fullhad qq z_qq z_tautau; do
        present=1
        for detector in "${detectors[@]}"; do
            [ -d "$POST/$detector/$sample" ] || present=0
        done
        if [ "$present" = 1 ]; then samples+=("$sample"); else echo "Skipping $sample: not converted for every detector" >&2; fi
    done
    [ "${#samples[@]}" -gt 0 ] || { echo "No sample is converted for all of: ${detectors[*]}" >&2; exit 1; }
    names=$(IFS=/; echo "${detectors[*]^^}")
    NOTE="$names; $EVENTS events per sample and detector, converted with the current postprocessor. Generator jets are clustered before detector visibility and target assignment; target jets contain the retained training targets. Samples are compared statistically: shared seed labels do not make the generator events identical, so check the generator spectra first. Targets carry truth kinematics, so a unit target/truth jet response is not a detector energy-resolution measurement."
    if [[ " ${detectors[*]} " == *" ild "* ]]; then NOTE+=" ILD dE/dx is not stored and is filled with zero."; fi
    ARGS=()
    for sample in "${samples[@]}"; do
        for detector in "${detectors[@]}"; do
            ARGS+=(--parquet "$sample-$detector:$detector=$POST/$detector/$sample")
        done
    done
    .venv/bin/python scripts/compare_detector_features.py "${ARGS[@]}" \
        --num-events "$EVENTS" --process "$(IFS=/; echo "${samples[*]}")" \
        --title "$names: all samples" --note "$NOTE" --output-dir "$OUTPUT/all_samples"
    for sample in "${samples[@]}"; do
        ARGS=()
        for detector in "${detectors[@]}"; do ARGS+=(--parquet "$detector=$POST/$detector/$sample"); done
        .venv/bin/python scripts/compare_detector_features.py "${ARGS[@]}" \
            --num-events "$EVENTS" --process "$sample" --title "$names: $sample" \
            --note "$NOTE" --output-dir "$OUTPUT/$sample"
    done
    exit 0
fi

WORK=local_test_data/detector_comparison
# Key4hep inputs are site-local: the ROOT files (<root>/<detector>/ttbar/) and stored TFDS
# version directories. Override with environment variables on another machine.
# The TFDS defaults are the Key4hep v1.4.0 recipe versions.
KEY4HEP_ROOT_DIR=${KEY4HEP_ROOT_DIR:-/mnt/work/mlpf/root}
CLIC_TFDS=${CLIC_TFDS:-/mnt/work/mlpf/tensorflow_datasets/clic/clic_edm_ttbar_pf/1/3.4.0}
CLD_TFDS=${CLD_TFDS:-/mnt/work/mlpf/tensorflow_datasets/cld/cld_edm_ttbar_pf/1/3.4.0}
IDEA_TFDS=${IDEA_TFDS:-/mnt/work/mlpf/tensorflow_datasets/idea/idea_edm_ttbar_pf/1/0.5.0}
ILD_TFDS=${ILD_TFDS:-/mnt/work/mlpf/tensorflow_datasets/ild/ild_edm_ttbar_pf/1/0.2.0}
# IDEA ROOT files from before the v1.4.0 production have no FittedTracks; convert those
# with IDEA_TRACK_SOURCE=truth_seeded. Each track source gets its own Parquet directory.
IDEA_TRACK_SOURCE=${IDEA_TRACK_SOURCE:-fitted}
case "$IDEA_TRACK_SOURCE" in
    fitted | truth_seeded) ;;
    *) echo "IDEA_TRACK_SOURCE must be fitted or truth_seeded, not '$IDEA_TRACK_SOURCE'" >&2; exit 1 ;;
esac
# Key4hep conversions are keyed by the converter sources, like the ColliderML build below,
# so a converter change never reuses stale Parquet files. Old conversions are preserved.
KEY4HEP_FINGERPRINT=$(sha256sum mlpf/data/key4hep/*.py mlpf/data/target_building.py mlpf/conf.py | sha256sum | cut -c1-12)
KEY4HEP_PARQUET="$WORK/key4hep_builds/$KEY4HEP_FINGERPRINT"
IDEA_PARQUET="$KEY4HEP_PARQUET/idea_$IDEA_TRACK_SOURCE"
# ColliderML and MAIA are built here at their builders' current versions.
builder_version() { sed -n 's/.*VERSION = tfds.core.Version(.*"\([0-9.]*\)").*/\1/p' "$1"; }
COLLIDERML_VERSION=$(builder_version mlpf/heptfds/colliderml_pf/ttbar.py)
MAIA_VERSION=$(builder_version mlpf/heptfds/maia_pf_edm4hep/ttbar.py)
# Isolate builds when converter/feature/TFDS logic changes. Old data is preserved.
COLLIDERML_FINGERPRINT=$(sha256sum mlpf/data/colliderml/{postprocessing,tracks,reader,truth,clustering}.py \
    mlpf/data/target_building.py mlpf/heptfds/colliderml_pf/ttbar.py \
    mlpf/heptfds/colliderml_utils/utils.py mlpf/conf.py | sha256sum | cut -c1-12)
COLLIDERML_BUILD="$WORK/colliderml_builds/$COLLIDERML_FINGERPRINT"
mkdir -p "$KEY4HEP_PARQUET"/{clic,cld,ild} "$IDEA_PARQUET"

if [ ! -f "$WORK/colliderml_source/provenance.json" ]; then
    HF_HOME="$PWD/$WORK/huggingface_cache" .venv/bin/python scripts/download_colliderml_comparison_sample.py \
        --output-dir "$WORK/colliderml_source"
fi

# ILD is optional: it is skipped when its ROOT file is not available on this machine.
KEY4HEP_PARQUET_ARGS=()
for detector in clic cld idea ild; do
    energy=365
    outpath="$KEY4HEP_PARQUET/$detector"
    extra_args=()
    if [ "$detector" = clic ]; then energy=380; fi
    if [ "$detector" = idea ]; then
        outpath="$IDEA_PARQUET"
        extra_args=(--idea-track-source "$IDEA_TRACK_SOURCE")
    fi
    filename="reco_p8_ee_ttbar_ecm${energy}_300000"
    if [ "$detector" = ild ] && [ ! -f "$outpath/$filename.parquet" ] && [ ! -f "$KEY4HEP_ROOT_DIR/ild/ttbar/$filename.root" ]; then
        echo "Skipping ild: $KEY4HEP_ROOT_DIR/ild/ttbar/$filename.root not found" >&2
        continue
    fi
    KEY4HEP_PARQUET_ARGS+=(--parquet "$detector=$outpath")
    if [ ! -f "$outpath/$filename.parquet" ]; then
        # Uproot 5.7.2's default FSSpecSource stalls on these files' PODIO string
        # metadata. Its local file reader succeeds; the physics conversion is unchanged.
        .venv/bin/python -u -c 'import functools, runpy, uproot; uproot.open = functools.partial(uproot.open, handler=uproot.source.file.MultithreadedFileSource); runpy.run_module("mlpf.data.key4hep.postprocessing", run_name="__main__")' \
            --input "$KEY4HEP_ROOT_DIR/$detector/ttbar/$filename.root" \
            --outpath "$outpath" --detector "$detector" --num-events 100 "${extra_args[@]}"
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

KEY4HEP_ROOT_DIR="$KEY4HEP_ROOT_DIR" \
COLLIDERML_DISPLAY_PARQUET="$COLLIDERML_BUILD/parquet/train-00000-of-01000.parquet" \
    bash scripts/run_detector_event_displays.sh

# Stored Key4hep TFDS columns are optional: skip any that are not available on this machine.
TFDS_ARGS=(--tfds "colliderml=$COLLIDERML_BUILD/tensorflow_datasets/colliderml_ttbar_nopu_pf/10/$COLLIDERML_VERSION")
for spec in "clic=$CLIC_TFDS" "cld=$CLD_TFDS" "idea=$IDEA_TFDS" "ild=$ILD_TFDS"; do
    if [ -f "${spec#*=}/dataset_info.json" ]; then
        TFDS_ARGS+=(--tfds "$spec")
    else
        echo "Skipping ${spec%%=*} TFDS: ${spec#*=} not found" >&2
    fi
done
TFDS_ARGS+=(--tfds "maia=$MAIA_TFDS")

.venv/bin/python scripts/compare_detector_features.py \
    --parquet "colliderml=$COLLIDERML_BUILD/parquet" \
    "${KEY4HEP_PARQUET_ARGS[@]}" \
    --parquet maia=local_test_data/maia_smoke/parquet \
    "${TFDS_ARGS[@]}" \
    --num-events 100 --split train+test --verify-colliderml-roundtrip --verify-maia-roundtrip \
    --maia-validation-report local_test_data/maia_smoke/validation_plots/validation_report.json \
    --output-dir "$WORK/real_plots"
