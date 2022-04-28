#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/grid.cern.ch/c7ui-test/etc/profile.d/setup-c7-ui-example.sh

cd ~/reco/mlpf/CMSSW_12_3_0_pre6
eval `scramv1 runtime -sh`

$CMSSW_BASE/src/Validation/RecoParticleFlow/test/run_relval.sh QCDPU reco 1
