#!/bin/bash

cmsDriver.py TTbar_14TeV_TuneCUETP8M1_cfi \
  --conditions auto:phase2_realistic \
  -n 10 \
  --era Phase2C8 \
  --eventcontent FEVTDEBUGHLT \
  -s GEN,SIM,DIGI:pdigi_valid,L1,L1TrackTrigger,DIGI2RAW,HLT:@fake2 \
  --datatier GEN-SIM \
  --beamspot NoSmear \
  --geometry Extended2026D41 \
  --pileup AVE_200_BX_25ns \
  --pileup_input das:/RelValMinBias_14TeV/CMSSW_10_6_0_patch2-106X_upgrade2023_realistic_v3_2023D41noPU-v1/GEN-SIM \
  --no_exec \
  --python_filename=step2.py

cmsDriver.py step3 \
  --conditions auto:phase2_realistic \
  -n 10 \
  --era Phase2C8 \
  --eventcontent FEVTDEBUGHLT,DQM \
  --runUnscheduled \
  -s RAW2DIGI,L1Reco,RECO,RECOSIM,VALIDATION:@phase2Validation,DQM:@phase2 \
  --datatier GEN-SIM-RECO,DQMIO \
  --geometry Extended2026D41 \
  --no_exec \
  --python_filename=RECO_fragment.py
