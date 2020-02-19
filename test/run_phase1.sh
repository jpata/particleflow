#!/bin/bash

#cmsDriver.py TTbar_14TeV_TuneCUETP8M1_cfi \
#  --conditions auto:phase1_2021_realistic \
#  -n 10 \
#  --era Run3 \
#  --eventcontent FEVTDEBUGHLT \
#  -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT \
#  --datatier GEN-SIM \
#  --geometry DB:Extended \
#  --pileup Run3_Flat55To75_PoissonOOTPU \
#  --pileup_input das:/RelValMinBias_14TeV/CMSSW_11_0_0_pre12-110X_mcRun3_2021_realistic_v5-v1/GEN-SIM \
#  --no_exec \
#  --python_filename=step2_phase1.py

cmsDriver.py step3 \
  --conditions auto:phase1_2021_realistic \
  -n 10 \
  --era Run3 \
  --eventcontent FEVTDEBUGHLT \
  --runUnscheduled \
  -s RAW2DIGI,L1Reco,RECO,RECOSIM \
  --datatier GEN-SIM-RECO \
  --geometry DB:Extended \
  --no_exec \
  --python_filename=RECO_fragment_phase1.py

