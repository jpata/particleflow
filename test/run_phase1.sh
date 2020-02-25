#!/bin/bash

set -e

#Generate the MC
#cmsDriver.py TTbar_14TeV_TuneCUETP8M1_cfi \
#  --conditions auto:phase1_2021_realistic \
#  -n 100 \
#  --era Run3 \
#  --eventcontent FEVTDEBUGHLT \
#  -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT \
#  --datatier GEN-SIM \
#  --geometry DB:Extended \
#  --pileup Run3_Flat55To75_PoissonOOTPU \
#  --pileup_input das:/RelValMinBias_14TeV/CMSSW_11_0_0_pre12-110X_mcRun3_2021_realistic_v5-v1/GEN-SIM \
#  --no_exec \
#  --fileout step2_phase1.root \
#  --customise RecoNtuples/HGCalAnalysis/step2_customise.customise \
#  --python_filename=step2_phase1.py
#
##Run the reco sequences
#cmsDriver.py step3 \
#  --conditions auto:phase1_2021_realistic \
#  --era Run3 \
#  --eventcontent FEVTDEBUGHLT \
#  --runUnscheduled \
#  -s RAW2DIGI,L1Reco,RECO,RECOSIM \
#  --datatier GEN-SIM-RECO \
#  --geometry DB:Extended \
#  --no_exec \
#  --filein step2_phase1.root \
#  --fileout step3_phase1.root \
#  --python_filename=step3_phase1.py

cmsRun step2_phase1.py &> log_step2.txt
cmsRun step3_phase1.py &> log_step3.txt
cmsRun CMSSW_11_0_0_pre12/src/RecoNtuples/HGCalAnalysis/test/pfntuple.py
