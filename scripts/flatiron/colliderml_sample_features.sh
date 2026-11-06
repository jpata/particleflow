#!/bin/bash
# Feature-comparison sheets across the ColliderML Release-1 samples (TFDS 1.5.0), one column per
# sample: one report for pu0, one for pu200, each with ttbar as the reference column. Plain bash
# from the repo root; reads N random events of one TFDS config per sample.
#
#   bash scripts/flatiron/colliderml_sample_features.sh [num_events] [outdir]
#
# -> <outdir>/{pu0,pu200}/index.html (+ per-group PNG sheets and summary.json)
set -euo pipefail

NUM_EVENTS="${1:-100}"
OUT="${2:-local_test_data/colliderml_sample_comparison}"
TFDS_DIR=/mnt/ceph/users/ewulff/tensorflow_datasets/colliderml
CONFIG=1
VERSION=1.5.0
export PYTHONPATH="$(pwd)"
export KERAS_BACKEND=torch

run() {
    local pu="$1" suffix="$2"
    shift 2
    local args=()
    for process in "$@"; do
        args+=(--tfds "${process}_${pu}:colliderml=${TFDS_DIR}/colliderml_${process}_${suffix}_pf/${CONFIG}/${VERSION}")
    done
    .venv/bin/python scripts/compare_detector_features.py "${args[@]}" \
        --num-events "${NUM_EVENTS}" --split test --process "ColliderML ${pu}" \
        --title "ColliderML Release-1 ${pu} sample feature comparison" \
        --note "ColliderML Release-1 pp samples at ${pu}, TFDS ${VERSION} config ${CONFIG} test split, ${NUM_EVENTS} random events each; ttbar is the reference. pileup_only counts every particle as pileup (ispu = pufrac = 1)." \
        --output-dir "${OUT}/${pu}"
}

run pu0 nopu ttbar dihiggs ggf higgs_portal hidden_valley zprime pileup_only
run pu200 pu200 ttbar dihiggs ggf zee zmumu
echo "done: ${OUT}/pu0/index.html ${OUT}/pu200/index.html"
