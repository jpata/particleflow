source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820
git clone https://github.com/jpata/particleflow
cd particleflow

scramv1 project CMSSW CMSSW_11_1_0_pre5
cd CMSSW_11_1_0_pre5/src
eval `scramv1 runtime -sh`
git cms-init

git remote add -f jpata https://github.com/jpata/cmssw
git fetch -a jpata

git cms-addpkg Validation/RecoParticleFlow
git cms-addpkg SimGeneral/CaloAnalysis/
git cms-addpkg SimGeneral/MixingModule/
git checkout -b jpata_pfntuplizer --track jpata/jpata_pfntuplizer

scram b
