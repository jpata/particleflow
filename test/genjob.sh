#!/bin/bash
set -e
set -x

WORKDIR=`pwd`

#seed must be greater than 0
SEED=$1

env
source /cvmfs/cms.cern.ch/cmsset_default.sh

CMSSWDIR=/storage/user/jpata/particleflow/test/CMSSW_11_0_0_pre12

cd $CMSSWDIR
eval `scramv1 runtime -sh`

cd $WORKDIR

mkdir testjob
cd testjob

cp $CMSSWDIR/../step2_phase1_new.py ./
cp $CMSSWDIR/../step3_phase1_new.py ./
cp $CMSSWDIR/../110X_mcRun3_2021_realistic_v8.db ./

pwd
ls -lrt

echo "process.RandomNumberGeneratorService.generator.initialSeed = $SEED" >> step2_phase1_new.py
cat step2_phase1_new.py
cmsRun step2_phase1_new.py
cmsRun step3_phase1_new.py
cmsRun $CMSSWDIR/src/Validation/RecoParticleFlow/test/pfanalysis_ntuple.py

cp pfntuple.root $CMSSWDIR/../pfntuple_$SEED.root
rm -Rf testjob
