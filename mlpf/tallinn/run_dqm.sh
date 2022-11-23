#!/bin/bash

find data/TTbarPU_baseline/ -name "*MINI_*.root" | sed "s/^/file:/" | sort -h > files_mini.txt
cmsDriver.py step5 --conditions auto:phase1_2021_realistic -s DQM:@pfDQM --datatier DQMIO --nThreads 8 --era Run3 --eventcontent DQM --filein filelist:files_mini.txt --fileout file:step5.root -n -1
cmsDriver.py step6 --conditions auto:phase1_2021_realistic -s HARVESTING:@pfDQM --era Run3 --filetype DQM --filein file:step5.root --fileout file:step6.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root data/TTbarPU_baseline/

find data/TTbarPU_mlpf/ -name "*MINI_*.root" | sed "s/^/file:/" | sort -h > files_mini.txt
cmsDriver.py step5 --conditions auto:phase1_2021_realistic -s DQM:@pfDQM --datatier DQMIO --nThreads 8 --era Run3 --eventcontent DQM --filein filelist:files_mini.txt --fileout file:step5.root -n -1
cmsDriver.py step6 --conditions auto:phase1_2021_realistic -s HARVESTING:@pfDQM --era Run3 --filetype DQM --filein file:step5.root --fileout file:step6.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root data/TTbarPU_mlpf/

find data/QCDPU_baseline/ -name "*MINI_*.root" | sed "s/^/file:/" | sort -h > files_mini.txt
cmsDriver.py step5 --conditions auto:phase1_2021_realistic -s DQM:@pfDQM --datatier DQMIO --nThreads 8 --era Run3 --eventcontent DQM --filein filelist:files_mini.txt --fileout file:step5.root -n -1
cmsDriver.py step6 --conditions auto:phase1_2021_realistic -s HARVESTING:@pfDQM --era Run3 --filetype DQM --filein file:step5.root --fileout file:step6.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root data/QCDPU_baseline/

find data/QCDPU_mlpf/ -name "*MINI_*.root" | sed "s/^/file:/" | sort -h > files_mini.txt
cmsDriver.py step5 --conditions auto:phase1_2021_realistic -s DQM:@pfDQM --datatier DQMIO --nThreads 8 --era Run3 --eventcontent DQM --filein filelist:files_mini.txt --fileout file:step5.root -n -1
cmsDriver.py step6 --conditions auto:phase1_2021_realistic -s HARVESTING:@pfDQM --era Run3 --filetype DQM --filein file:step5.root --fileout file:step6.root
cp DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root data/QCDPU_mlpf/
