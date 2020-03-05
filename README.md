Notes on modernizing CMS particle flow, in particular [PFBlockAlgo](https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/src/PFBlockAlgo.cc) and [PFAlgo](https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/src/PFAlgo.cc).

# Overview

- [x] set up datasets and ntuples for detailed PF analysis
  - [x] simple python version in [test/ntuplizer.py](test/ntuplizer.py)
  - [ ] advanced CMSSW version with generator truth in [Validation/RecoParticleFlow/PFAnalysis.cc](https://github.com/jpata/cmssw/blob/jpata_pfntuplizer/Validation/RecoParticleFlow/plugins/PFAnalysis.cc)
- [ ] GPU code for existing PF algorithms
  - [x] test CLUE for element to block clustering
  - [ ] port CLUE to PFBlockAlgo in CMSSW
  - [ ] parallelize PFAlgo calls on blocks
  - [ ] GPU-implementation of PFAlgo
- [ ] reproduce existing PF with machine learning
  - [x] test element-to-block clustering with ML (Edge classifier, GNN)
  - [x] test block-to-candidate regression
  - [ ] end-to-end training of elements to MLPF-candidates using GNN-s
    - [x] first baseline training converges to multiclass accuracy > 0.96, momentum correlation > 0.9
    - [ ] improve training speed
    - [ ] detailed hyperparameter scan
    - [ ] further reduce bias in end-to-end training (muons, electrons, momentum tails)
- [ ] reconstruct genparticles directly from detector elements a la HGCAL, neutrino experiments etc
  - [x] set up datasets for regression genparticles from elements
    - [ ] develop improved loss function for event-to-event comparison: EMD, GAN

## Presentations

- ML4RECO meeting, 2020-03-04: https://indico.cern.ch/event/895228/contributions/3776739/attachments/1998928/3335497/2020_03_04.pdf
- CMS PF group, 2020-02-28: https://indico.cern.ch/event/892992/contributions/3766807/attachments/1995348/3328771/2020_02_28.pdf
- CMS PF group, 2020-01-31: https://indico.cern.ch/event/885043/contributions/3730304/attachments/1979098/3295074/2020_01_30_pf.pdf
- FNAL HGCAL ML meeting, 2020-01-30: https://indico.cern.ch/event/884801/contributions/3730336/attachments/1978912/3294638/2020_01_30.pdf
- Caltech group meeting, 2020-01-28: https://indico.cern.ch/event/881683/contributions/3714961/attachments/1977131/3291096/2020_01_21.pdf
- CMS PF group, 2020-01-17: https://indico.cern.ch/event/862200/contributions/3706909/attachments/1971145/3279010/2020_01_16.pdf
- CMS PF group, 2019-11-22: https://indico.cern.ch/event/862195/contributions/3649510/attachments/1949957/3236487/2019_11_22.pdf
- CMS PF group, 2019-11-08: https://indico.cern.ch/event/861409/contributions/3632204/attachments/1941376/3219105/2019_11_08.pdf
- Caltech ML meeting, 2019-10-31: https://indico.cern.ch/event/858644/contributions/3623446/attachments/1936711/3209684/2019_10_07_pf.pdf
- Caltech ML meeting, 2019-09-19: https://indico.cern.ch/event/849944/contributions/3572113/attachments/1911520/3158764/2019_09_18_pf_ml.pdf
- CMS PF group, 2019-09-10: https://indico.cern.ch/event/846887/contributions/3557300/attachments/1904664/3145310/2019_09_10_pf_refactoring.pdf
- Caltech ML meeting, 2019-09-05: https://indico.cern.ch/event/845349/contributions/3554787/attachments/1902837/3141723/2019_09_05_pfalgo.pdf

## Other relevant issues, repos, PR-s:

- https://github.com/jpata/cmssw/issues/56

## Setting up the code

From [setup.sh](test/setup.sh):

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820
git clone https://github.com/jpata/particleflow
cd particleflow

scramv1 project CMSSW CMSSW_11_0_1
cd CMSSW_11_0_1/src
eval `scramv1 runtime -sh`
git cms-init

git remote add -f jpata https://github.com/jpata/cmssw
git fetch -a jpata

git cms-addpkg Validation/RecoParticleFlow
git cms-addpkg SimGeneral/CaloAnalysis/
git checkout -b jpata_pfntuplizer --track jpata/jpata_pfntuplizer

scram b
cd ../..
```

## Datasets
- February 2020
  - /RelValQCD_FlatPt_15_3000HS_14/CMSSW_11_0_0_pre12-PU_110X_mcRun3_2021_realistic_v5-v1/GEN-SIM-DIGI-RAW
    - EDM: `/mnt/hadoop/store/user/jpata/RelValQCD_FlatPt_15_3000HS_14/pfvalidation/191126_233511/0000/step3_AOD*.root`
    - flat ROOT: `/storage/user/jpata/particleflow/data/QCD_run3/step3_ntuple_*.root` or `/eos/user/j/jpata/particleflow/QCD_run3/step3_AOD*.root`
    - npy: `/storage/user/jpata/particleflow/data/QCD_run3/step3_ntuple_*.npz`
  - /RelValTTbar_14TeV/CMSSW_11_0_0_pre12-PU_110X_mcRun3_2021_realistic_v5-v1/GEN-SIM-DIGI-RAW
    - EDM: `/mnt/hadoop/store/user/jpata/RelValTTbar_14TeV/pfvalidation/191126_233751/0000/step3_AOD*.root`
    - flat ROOT: `/storage/user/jpata/particleflow/data/TTbar_run3/step3_ntuple_*.root` or `/eos/user/j/jpata/particleflow/TTbar_run3/step3_AOD*.root`
    - npy: `/storage/user/jpata/particleflow/data/TTbar_run3/step3_ntuple_*.npz`

## Running step3 on grid

PF reconstruction from `GEN-SIM-DIGI-RAW` needs to be run on the grid:
```bash
#Run the crab jobs
cd test
edmConfigDump step3.py > step3_dump.py
python multicrab.py
```

## Contents of the flat ROOT output ntuple

Produced using
```bash
python3 test/ntuplizer.py ./data/QCD_run3 /mnt/hadoop/store/user/jpata/RelValQCD_FlatPt_15_3000HS_14/pfvalidation/191126_233511/0000/step3_AOD_1.root
```

The TTree `pftree` contains the elements, candidates and genparticles:
- [PFCluster](https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowReco/interface/PFCluster.h)
  - `PFBlockElement::(ECAL, PS1, PS2, HCAL, GSF, HO, HFHAD, HFEM)`           
- [PFRecTrack](https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowReco/interface/PFRecTrack.h) from
  - `PFBlockElement::TRACK` or `PFBlockElement::BREM`
- [PFCandidates](https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/interface/PFCandidate.h)
- genparticles

## Numpy training ntuples

Produced using

```bash
python3 test/graph.py ./data/QCD_run3/step3_AOD_1.root
```
- `step3_AOD_1_ev.npz`: PF elements, candidates, and the block associations via a numerical ID
  - `elements: [Nelem, Nelem_features]` for the input PFElement data
  - `element_block_id: [Nelem, ]` for the PFAlgo-based block id
  - `candidates: [Ncand, Ncand_features]` for the output PFCandidate data
  - `candidate_block_id: [Ncand, ]` for the PFAlgo-based block id 
- `step3_AOD_1_dist.npz`: sparse `[Nelem, Nelem]` distance matrix from PFBlockAlgo between the candidates

## Model training

On iBanks:
```bash
singularity exec --nv -B /storage ~jpata/gpuservers/singularity/images/pytorch.simg python3 \
    test/train_end2end.py --model PFNet6 --n_train 200 \
    --batch_size 2 --n_epoch 100 --lr 0.0001 --hidden_dim 512
```

## Model validation

Notebook: [test_end2end](notebooks/test_end2end.ipynb)

## Acknowledgements

Part of this work was conducted at **iBanks**, the AI GPU cluster at Caltech. We acknowledge NVIDIA, SuperMicro and the Kavli Foundation for their support of **iBanks**.
