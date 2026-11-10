#!/bin/bash
# Download, convert and TFDS-build ColliderML Release-1 samples end to end, as one chain of
# Slurm jobs per sample. Run from the repo root with plain bash (NOT sbatch); it only submits:
#
#   bash scripts/flatiron/colliderml_process_samples.sh --dry-run   # print the sbatch commands
#   bash scripts/flatiron/colliderml_process_samples.sh             # all non-ttbar samples
#   bash scripts/flatiron/colliderml_process_samples.sh zee_pu200 ggf_pu0   # a subset
#   bash scripts/flatiron/colliderml_process_samples.sh --lanes 2 --dl-parallel 2   # more bandwidth
#
# Per sample <process>_<pileup>:
#   1. colliderml_download.sh   --array=0-3%N one task per object table, N at a time (genx, 8 CPUs)
#   2. colliderml_convert.sh    one whole gen node, 64 ranks over shards 0-1000 (afterok:1)
#      -> mlpf_parquet/clue/<sample>/
#   3. colliderml_tfds.sh       --array=1-10  one task per TFDS config (afterok:2)
#      -> tensorflow_datasets/colliderml/colliderml_<process>_{nopu,pu200}_pf/<config>/1.5.0
# Downloads run in --lanes parallel chains (default 1): sample i waits (afterany) for sample
# i-LANES, and each sample downloads --dl-parallel tables at once (default 1), so at most
# LANES x DL_PARALLEL download streams run. Keep the defaults: SCC asked us (2026-10-30) to stay
# within 1-2 Gbps of internet bandwidth after 12 streams reached 5-8 Gbps on one node; the
# download script also caps hf_xet's per-process concurrency. More streams mostly turn into
# HTTP 429 retries anyway. Conversions and TFDS builds of different samples overlap freely.
#
# Every step is resumable: downloads skip complete files, the converter skips finished shards,
# TFDS skips prepared configs. If a step fails, its dependents stay pending with reason
# DependencyNeverSatisfied: scancel them and rerun this script for that sample.

set -euo pipefail

DRY_RUN=0
LANES=1
DL_PARALLEL=1
SAMPLES=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --lanes)
            if [ "$#" -lt 2 ]; then echo "ERROR: --lanes needs a value" >&2; exit 2; fi
            LANES="$2"
            shift
            ;;
        --dl-parallel)
            if [ "$#" -lt 2 ]; then echo "ERROR: --dl-parallel needs a value" >&2; exit 2; fi
            DL_PARALLEL="$2"
            shift
            ;;
        *) SAMPLES+=("$1") ;;
    esac
    shift
done
if ! [[ "${LANES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --lanes needs a positive integer, got '${LANES}'" >&2
    exit 2
fi
if ! [[ "${DL_PARALLEL}" =~ ^[1-4]$ ]]; then
    echo "ERROR: --dl-parallel needs an integer 1-4, got '${DL_PARALLEL}'" >&2
    exit 2
fi
if [ "${#SAMPLES[@]}" -eq 0 ]; then
    # every Release-1 sample at the pinned revision except ttbar (already processed)
    SAMPLES=(
        dihiggs_pu0 ggf_pu0 higgs_portal_pu0 hidden_valley_pu0 zprime_pu0 pileup_only_pu0
        zee_pu200 zmumu_pu200 ggf_pu200 dihiggs_pu200
    )
fi

if [ ! -f scripts/flatiron/colliderml_download.sh ]; then
    echo "ERROR: run from the repo root" >&2
    exit 2
fi
mkdir -p logs_slurm

for SAMPLE in "${SAMPLES[@]}"; do
    PROCESS="${SAMPLE%_*}"
    case "${SAMPLE##*_}" in
        pu0) BUILDER="mlpf/heptfds/colliderml_pf/${PROCESS}.py" ;;
        pu200) BUILDER="mlpf/heptfds/colliderml_pf/${PROCESS}_pu200.py" ;;
        *) echo "ERROR: sample '${SAMPLE}' is not <process>_<pu0|pu200>" >&2; exit 2 ;;
    esac
    if [ ! -f "${BUILDER}" ]; then
        echo "ERROR: no TFDS builder ${BUILDER} for ${SAMPLE}" >&2
        exit 2
    fi
done

# submit one job; prints the job id (a placeholder in dry-run mode)
submit() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "sbatch --parsable $*" >&2
        echo "<${JOB_TAG}>"
    else
        sbatch --parsable "$@"
    fi
}

DL_IDS=()
for SAMPLE in "${SAMPLES[@]}"; do
    # chain onto the download LANES samples back (none for the first LANES samples)
    PREV_DL=""
    if [ "${#DL_IDS[@]}" -ge "${LANES}" ]; then
        PREV_DL="${DL_IDS[$(( ${#DL_IDS[@]} - LANES ))]}"
    fi
    JOB_TAG="${SAMPLE}_download"
    DL=$(submit --job-name="cml_dl_${SAMPLE}" --array=0-3%${DL_PARALLEL} ${PREV_DL:+--dependency=afterany:${PREV_DL}} \
        scripts/flatiron/colliderml_download.sh "${SAMPLE}")
    JOB_TAG="${SAMPLE}_convert"
    CONV=$(submit --job-name="cml_conv_${SAMPLE}" --dependency="afterok:${DL}" \
        scripts/flatiron/colliderml_convert.sh 0 1000 "${SAMPLE}")
    JOB_TAG="${SAMPLE}_tfds"
    TFDS=$(submit --job-name="cml_tfds_${SAMPLE}" --array=1-10 --dependency="afterok:${CONV}" \
        scripts/flatiron/colliderml_tfds.sh "${SAMPLE}")
    echo "${SAMPLE}: download ${DL} -> convert ${CONV} -> tfds ${TFDS}"
    DL_IDS+=("${DL}")
done
