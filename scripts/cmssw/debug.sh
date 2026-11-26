#!/usr/bin/env bash
# Isolated, logged experiments using env.sh's exact runtime.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
events=
while (($#)); do
    case $1 in
        --events)
            [[ $# -ge 2 && $2 =~ ^(-1|[0-9]+)$ ]] || { echo 'Expected --events N (N >= 0, or -1 for all)' >&2; exit 2; }
            events=$2; shift 2 ;;
        --help)
            echo 'Usage: scripts/cmssw/debug.sh [--events N] [-- COMMAND [ARG...]]'
            echo 'Default: isolated step2 cmsRun. Logs and outputs: logs/cmssw/<timestamp>.<id>/'
            exit 0 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
if [[ -n $events && $# -gt 0 ]]; then
    echo '--events applies only to the default step2 command.' >&2
    exit 2
fi
release_dir=${CMSSW_RELEASE_DIR:-$repo_dir/CMSSW_20_1_0_pre5}
job_dir=${CMSSW_JOB_DIR:-$release_dir/34506.213_SinglePiFlatPt0p7To10+Run4D121_HGCALNanoVal}
job_dir=$(cd -- "$job_dir" && pwd)
mkdir -p "$repo_dir/logs/cmssw"
run_dir=$(mktemp -d "$repo_dir/logs/cmssw/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
# Snapshot configs, link existing ROOT inputs, and keep new outputs in this run.
cp -- "$job_dir/"*.py "$run_dir/"
for input in "$job_dir/"*.root; do
    [[ -e $input ]] || continue
    # step2.root is this job's output, never link it back into the original job.
    [[ $(basename -- "$input") == step1.root ]] || continue
    ln -s -- "$input" "$run_dir/$(basename -- "$input")"
done
if (($# == 0)); then
    if [[ -n $events ]]; then
        cat > "$run_dir/debug_step2.py" <<'PY'
import os
import runpy
from pathlib import Path
config = Path(__file__).resolve().with_name("step2_DIGI_L1TrackTrigger_L1_L1P2GT_DIGI2RAW_HLT.py")
process = runpy.run_path(str(config))["process"]
process.maxEvents.input = int(os.environ["PF_CMSSW_MAX_EVENTS"])
PY
        export PF_CMSSW_MAX_EVENTS=$events
        set -- cmsRun "$run_dir/debug_step2.py"
    else
        set -- cmsRun "$run_dir/step2_DIGI_L1TrackTrigger_L1_L1P2GT_DIGI2RAW_HLT.py"
    fi
fi
{
    printf 'UTC: %s\nRelease: %s\nJob: %s\nRun: %s\n' "$(date -u +%FT%TZ)" "$release_dir" "$job_dir" "$run_dir"
    printf 'Command: '; printf '%q ' "$@"; printf '\n'
    printf 'Event limit: %s\n' "${events:-configuration default}"
    printf 'SCRAM_ARCH: %s\nAPPTAINER_NV: %s\nUNPACKED_IMAGE: %s\n' "${SCRAM_ARCH:-el9_amd64_gcc14}" "${APPTAINER_NV:-unset}" "${UNPACKED_IMAGE:-launcher default}"
    printf 'SITECONFIG_PATH: %s\n' "${SITECONFIG_PATH:-CMS default}"
} > "$run_dir/command.txt"
echo "Run directory: $run_dir"
set +e
CMSSW_RUN_DIR=$run_dir "$repo_dir/env.sh" "$@" 2>&1 | tee "$run_dir/run.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$run_dir/exit_code"
echo "Exit status: $status; log: $run_dir/run.log"
exit "$status"
