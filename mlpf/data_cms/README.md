## Validation data

### ACAT 2022
The MLPF CMSSW results presented at ACAT can be reproduced using the MINIAOD samples from
```
gfal-copy -r root://xrootd.hep.kbfi.ee:1094//store/user/jpata/mlpf/results/acat2022 ./
```
See below for the steps to reproduce these samples.

The resulting plots can be found at:
```
https://jpata.web.cern.ch/jpata/mlpf/results/acat2022_20221004_model40M_revalidation20240523/
```

## Code setup

The following should work on lxplus.
```
#ensure proxy is set
voms-proxy-init -voms cms -valid 192:00
voms-proxy-info

#Initialize SLC7
cmssw-el7

export SCRAM_ARCH=slc7_amd64_gcc10
cmsrel CMSSW_12_3_0_pre6
cd CMSSW_12_3_0_pre6/src
cmsenv
git cms-init

#checkout the MLPF code
git-cms-merge-topic jpata:pfanalysis_caloparticle

#check out the version from the 2022 release
git checkout mlpf_acat2022

#compile
scram b -j4

#download the MLPF model
mkdir -p src/RecoParticleFlow/PFProducer/data/mlpf/
wget https://huggingface.co/jpata/particleflow/resolve/main/cms/acat2022_20221004_model40M/dev.onnx?download=true -O RecoParticleFlow/PFProducer/data/mlpf/dev.onnx

# must be b786aa6de49b51f703c87533a66326d6
md5sum RecoParticleFlow/PFProducer/data/mlpf/dev.onnx
```

## Running MLPF in CMSSW
MLPF is integrated in CMSSW reconstruction and can be run either using simple but slow matrix workflows, or using the faster but more elaborate PF validation.

### Matrix workflows

Matrix workflows allow to run MLPF directly out of the box, rerunning the full reconstruction chain.
This is easy to run, but time consuming.
```
#check the workflows with the .13 suffix (that have MLPF enabled)
> runTheMatrix.py --what upgrade -n | grep "\.13"

#Run this workflow TTbar_14TeV + 2021PU_mlpf
> runTheMatrix.py --what upgrade -l 11834.13

#Takes around 30 minutes
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU Step0-PASSED Step1-PASSED Step2-PASSED Step3-PASSED  - time date Thu May 16 15:24:47 2024-date Thu May 16 15:06:24 2024; exit: 0 0 0 0
1 1 1 1 tests passed, 0 0 0 0 failed
```

Check the outputs
```
> ls 11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/*.root
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/histProbFunction.root
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/step1.root
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/step2.root
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/step3_inDQM.root
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/step3_inMINIAODSIM.root
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/step3_inNANOEDMAODSIM.root
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/step3_inRECOSIM.root
11834.13_TTbar_14TeV+2021PU_mlpf+TTbar_14TeV_TuneCP5_GenSim+DigiPU+RecoNanoPU+HARVESTNanoPU/step3.root
```

The particle flow candidates can be found in `step3.root`:
```
vector<reco::PFCandidate>             "particleFlow"              ""                "RECO"
```

### PF validation
To test MLPF on higher statistics, it's not practical to redo full reconstruction before the particle flow step.
We can follow a similar logic as the PF validation, where only the relevant PF sequences are rerun.

#### MINIAOD with PF and MLPF
The PF validation workflows can be run using the scripts in
```
cd particleflow

#the number 1 signifies the row index (filename) in the input file to process
./scripts/cmssw/validation_job.sh mlpf scripts/cmssw/qcd_pu.txt QCD_PU 1
./scripts/cmssw/validation_job.sh pf scripts/cmssw/qcd_pu.txt QCD_PU 1
```
Note: the input dataset is only stored on `T2_EE_Estonia`, therefore depending on grid accessibility, it might be slow to access.

The MINIAOD output will be in `$CMSSW_BASE/out/QCD_PU_mlpf` and `$CMSSW_BASE/out/QCD_PU_pf`.

#### DQM plots
Now the MINIAOD output can be analyzed with the DQM and PF validation scripts:
```
./scripts/cmssw/run_dqm.sh $CMSSW_BASE/out
```

The outputs will be in:
```
ls plots
```
and can be displayed in a web browser.

## Generating MLPF training samples
TODO.
