#!/bin/bash
set -e
set -x

CMSSWDIR=/storage/user/jpata/particleflow/test/CMSSW_11_1_0_pre5

WORKDIR=`pwd`

#sleep randomly up to 120s to stagger job start times
sleep $((RANDOM % 120))

#seed must be greater than 0
SAMPLE=$1
SEED=$2

PILEUP=NoPileUp
#PILEUP=Run3_Flat55To75_PoissonOOTPU
PILEUP_INPUT=filelist:/storage/user/jpata/particleflow/test/pu_files.txt
#--pileup_input $PILEUP_INPUT \

N=100

env
source /cvmfs/cms.cern.ch/cmsset_default.sh


cd $CMSSWDIR
eval `scramv1 runtime -sh`

cd $WORKDIR

mkdir testjob
cd testjob

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
cmsRun step2_phase1_new.py
cmsRun step3_phase1_new.py
cmsRun $CMSSWDIR/src/Validation/RecoParticleFlow/test/pfanalysis_ntuple.py

#cp *.root $CMSSWDIR/../$SAMPLE/
cp pfntuple.root $CMSSWDIR/../$SAMPLE/pfntuple_$SEED.root

rm -Rf testjob
