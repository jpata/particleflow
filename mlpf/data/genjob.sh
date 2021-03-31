#!/bin/bash
set -e
set -x

CMSSWDIR=/home/joosep/particleflow/mlpf/data/CMSSW_11_3_0_pre4

WORKDIR=$(mktemp -d -t job-XXXXXXXXXX --tmpdir=`pwd`)

#sleep randomly up to 120s to stagger job start times
#sleep $((RANDOM % 120))

#seed must be greater than 0
SAMPLE=$1
SEED=$2

PILEUP=NoPileUp

N=1000

env
source /cvmfs/cms.cern.ch/cmsset_default.sh


cd $CMSSWDIR
eval `scramv1 runtime -sh`

cd $WORKDIR

#Generate the MC
cmsDriver.py $SAMPLE \
  --conditions auto:phase1_2021_realistic \
  -n $N \
  --era Run3 \
  --eventcontent FEVTDEBUGHLT \
  -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT \
  --datatier GEN-SIM \
  --geometry DB:Extended \
  --pileup $PILEUP \
  --no_exec \
  --fileout step2_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step2 \
  --python_filename=step2_phase1_new.py

#Run the reco sequences
cmsDriver.py step3 \
  --conditions auto:phase1_2021_realistic \
  --era Run3 \
  -n -1 \
  --eventcontent FEVTDEBUGHLT \
  --runUnscheduled \
  -s RAW2DIGI,L1Reco,RECO,RECOSIM \
  --datatier GEN-SIM-RECO \
  --geometry DB:Extended \
  --no_exec \
  --filein file:step2_phase1_new.root \
  --fileout step3_phase1_new.root \
  --customise Validation/RecoParticleFlow/customize_pfanalysis.customize_step3 \
  --python_filename=step3_phase1_new.py

pwd
ls -lrt

echo "process.RandomNumberGeneratorService.generator.initialSeed = $SEED" >> step2_phase1_new.py
cmsRun step2_phase1_new.py &> step2.log
cmsRun step3_phase1_new.py &> step3.log
cmsRun $CMSSWDIR/src/Validation/RecoParticleFlow/test/pfanalysis_ntuple.py &> step4.log

#cp *.root $CMSSWDIR/../$SAMPLE/
cp pfntuple.root $CMSSWDIR/../$SAMPLE/pfntuple_$SEED.root
rm -Rf $WORKDIR
