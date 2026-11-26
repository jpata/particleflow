#!/usr/bin/env bash
# Internal container entry point; invoked by env.sh.
set -eo pipefail
export SCRAM_ARCH=$PF_CMSSW_ARCH
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd -- "$PF_CMSSW_RELEASE_DIR"
# Capture scram's status before eval so a setup failure cannot be hidden.
runtime=$(scramv1 runtime -sh)
eval "$runtime"
cd -- "$PF_CMSSW_RUN_DIR"
eval "exec $PF_CMSSW_COMMAND"
