#!/usr/bin/env bash
# Run any command in the same EL9/SCRAM environment as the failing step2 job.
set -eo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
Usage: ./env.sh [COMMAND [ARG...]]
No command: run the original step2 cmsRun job.
Examples:
  ./env.sh python3 --version
  ./env.sh bash -c 'echo "$CMSSW_BASE"; command -v cmsRun'
  ./env.sh bash -i
Overrides: CMSSW_RELEASE_DIR, CMSSW_JOB_DIR, SCRAM_ARCH, CMSSW_RUN_DIR.
Use scripts/cmssw/debug.sh for isolated runs with captured logs.
HELP
    exit 0
fi
export PF_CMSSW_RELEASE_DIR=${CMSSW_RELEASE_DIR:-$repo_dir/CMSSW_20_1_0_pre5}
export PF_CMSSW_JOB_DIR=${CMSSW_JOB_DIR:-$PF_CMSSW_RELEASE_DIR/34506.213_SinglePiFlatPt0p7To10+Run4D121_HGCALNanoVal}
export PF_CMSSW_RUN_DIR=${CMSSW_RUN_DIR:-$PF_CMSSW_JOB_DIR}
export PF_CMSSW_ARCH=${SCRAM_ARCH:-el9_amd64_gcc14}
export PF_CMSSW_ENTRY=$repo_dir/scripts/cmssw/inside.sh
if (($# == 0)); then
    set -- cmsRun step2_DIGI_L1TrackTrigger_L1_L1P2GT_DIGI2RAW_HLT.py
fi
# Serialize arguments with bash quoting; the container evaluates only this value.
printf -v PF_CMSSW_COMMAND '%q ' "$@"
export PF_CMSSW_COMMAND
launcher=$(command -v cmssw-el9 || true)
launcher=${launcher:-/cvmfs/cms.cern.ch/common/cmssw-el9}
if [[ ! -x $launcher ]]; then
    echo "Cannot find cmssw-el9; mount /cvmfs/cms.cern.ch first." >&2
    exit 127
fi
exec "$launcher" --command-to-run 'exec bash "$PF_CMSSW_ENTRY"'
