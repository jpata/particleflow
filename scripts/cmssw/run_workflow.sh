#!/usr/bin/env bash
# Run the full ten-event workflow in a new directory, preserving earlier runs.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
release_dir=${CMSSW_RELEASE_DIR:-$repo_dir/CMSSW_20_1_0_pre5}
job_dir=${CMSSW_JOB_DIR:-$release_dir/34506.213_SinglePiFlatPt0p7To10+Run4D121_HGCALNanoVal}
export APPTAINER_NV=${APPTAINER_NV:-true}
export SITECONFIG_PATH=${SITECONFIG_PATH:-/cvmfs/cms.cern.ch/SITECONF/T2_EE_Estonia}
mkdir -p "$repo_dir/logs/cmssw"
run_dir=$(mktemp -d "$repo_dir/logs/cmssw/$(date -u +%Y%m%dT%H%M%SZ)-10events.XXXXXX")
cp -- "$job_dir/SinglePiFlatPt0p7To10_cfi_GEN_SIM.py" "$job_dir/step"{2,3,4,5}_*.py "$run_dir/"
{
    printf 'UTC: %s\nRelease: %s\nJob: %s\nRun: %s\n' "$(date -u +%FT%TZ)" "$release_dir" "$job_dir" "$run_dir"
    printf 'Events: 10 (original configs); harvesting retains its configured limit\n'
    printf 'APPTAINER_NV: %s\nSITECONFIG_PATH: %s\nSCRAM_ARCH: %s\n' "$APPTAINER_NV" "$SITECONFIG_PATH" "${SCRAM_ARCH:-el9_amd64_gcc14}"
    sha256sum "$run_dir/"*.py
} > "$run_dir/workflow_command.txt"
echo "Run directory: $run_dir"
set +e
CMSSW_RUN_DIR=$run_dir "$repo_dir/env.sh" bash "$repo_dir/scripts/cmssw/run_remaining.sh" 1 2>&1 | tee "$run_dir/workflow.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$run_dir/workflow_exit_code"
echo "Workflow exit status: $status; run directory: $run_dir"
exit "$status"
