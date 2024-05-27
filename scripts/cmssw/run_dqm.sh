#!/bin/bash

OUTPUT_PATH=$1

#Create DQM outputs from MINIAOD
find $OUTPUT_PATH/QCD_PU_pf/ -name "*MINI_*.root" | sed "s/^/file:/" | sort -h > files_mini.txt
cmsDriver.py step5 --conditions auto:phase1_2023_realistic -s DQM:@pfDQM --datatier DQMIO --nThreads 8 --era Run3 --eventcontent DQM --filein filelist:files_mini.txt --fileout file:step5.root -n -1
cmsDriver.py step6 --conditions auto:phase1_2023_realistic -s HARVESTING:@pfDQM --era Run3 --filetype DQM --filein file:step5.root --fileout file:step6.root
mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root $OUTPUT_PATH/QCD_PU_pf/

find $OUTPUT_PATH/QCD_PU_mlpf/ -name "*MINI_*.root" | sed "s/^/file:/" | sort -h > files_mini.txt
cmsDriver.py step5 --conditions auto:phase1_2023_realistic -s DQM:@pfDQM --datatier DQMIO --nThreads 8 --era Run3 --eventcontent DQM --filein filelist:files_mini.txt --fileout file:step5.root -n -1
cmsDriver.py step6 --conditions auto:phase1_2023_realistic -s HARVESTING:@pfDQM --era Run3 --filetype DQM --filein file:step5.root --fileout file:step6.root
mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root $OUTPUT_PATH/QCD_PU_mlpf/

#Run DQM plots
PYTHONPATH=$CMSSW_BASE/src/Validation/RecoParticleFlow/test:$PYTHONPATH python3 scripts/cmssw/compare.py -s QCD:$OUTPUT_PATH/QCD_PU_pf/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root:$OUTPUT_PATH/QCD_PU_mlpf/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root --doResponsePlots --doOffsetPlots --doMETPlots --doPFCandPlots
