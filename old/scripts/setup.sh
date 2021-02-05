source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820

scramv1 project CMSSW CMSSW_11_1_0_pre5
cd CMSSW_11_1_0_pre5/src
eval `scramv1 runtime -sh`
git cms-init

git remote add -f jpata https://github.com/jpata/cmssw
git fetch -a jpata

git cms-addpkg RecoParticleFlow/PFProducer
git cms-addpkg Validation/RecoParticleFlow
git cms-addpkg SimGeneral/CaloAnalysis/
git cms-addpkg SimGeneral/MixingModule/

git checkout -b jpata_pfntuplizer --track jpata/jpata_pfntuplizer

#just to get an exact version of the code
git checkout 0fdcc0e8b6d848473170f0dc904468fa8a953aa8

#download the MLPF weight file
mkdir -p RecoParticleFlow/PFProducer/data/mlpf/
wget http://login-1.hep.caltech.edu/~jpata/particleflow/2020-05/models/mlpf_2020_05_19.pb -O RecoParticleFlow/PFProducer/data/mlpf/mlpf_2020_05_19.pb

scram b

#Run a small test of ML-PF
cmsRun RecoParticleFlow/PFProducer/test/mlpf_producer.py
edmDumpEventContent test.root | grep -i mlpf

#Run ML-PF within the reco framework up to ak4PFJets / ak4MLPFJets
cmsDriver.py step3 --runUnscheduled --conditions auto:phase1_2021_realistic \
  -s RAW2DIGI,L1Reco,RECO,RECOSIM,EI,PAT \
  --datatier MINIAODSIM --nThreads 1 -n 10 --era Run3 \
  --eventcontent MINIAODSIM --geometry=DB.Extended \
  --filein /store/relval/CMSSW_11_0_0_patch1/RelValQCD_FlatPt_15_3000HS_14/GEN-SIM-DIGI-RAW/PU_110X_mcRun3_2021_realistic_v6-v1/20000/087F3A84-A56F-784B-BE13-395D75616CC5.root \
  --customise RecoParticleFlow/PFProducer/mlpfproducer_customize.customize_step3 \
  --fileout file:step3_inMINIAODSIM.root
