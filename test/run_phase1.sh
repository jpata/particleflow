#!/bin/bash

set -e

SAMPLE=SingleElectronFlatPt1To100_pythia8_cfi
PILEUP=NoPileUp
PILEUP_INPUT=
#SAMPLE=TTbar_14TeV_TuneCUETP8M1_cfi
#PILEUP=Run3_Flat55To75_PoissonOOTPU
#PILEUP_INPUT=das:/RelValMinBias_14TeV/CMSSW_11_0_0_pre12-110X_mcRun3_2021_realistic_v5-v1/GEN-SIM
N=1

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

#cmsRun step2_phase1_new.py &> log_step2.txt
#cmsRun step3_phase1_new.py &> log_step3.txt
#cmsRun CMSSW_11_0_0_pre12/src/Validation/RecoParticleFlow/test/pfanalysis_ntuple.py
