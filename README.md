Notes on modernizing CMS particle flow, in particular [PFBlockAlgo](https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/src/PFBlockAlgo.cc) and [PFAlgo](https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/src/PFAlgo.cc).

# Overview

- [x] set up datasets and ntuples for detailed PF analysis
  - [x] advanced CMSSW version with generator truth in [Validation/RecoParticleFlow/PFAnalysis.cc](https://github.com/jpata/cmssw/blob/jpata_pfntuplizer/Validation/RecoParticleFlow/plugins/PFAnalysis.cc)
- [x] reproduce existing PFCandidates with machine learning
  - [x] end-to-end training of elements to MLPF-candidates using GNN-s
- [ ] reconstruct genparticles directly from detector elements a la HGCAL, neutrino experiments etc
  - [x] set up datasets for regression genparticles from elements
  - [x] Develop a baseline ML-PF model that is able to regress pions and neutral hadrons (currently GravNet-512 with 2D radius-graph)
  - [ ] develop improved loss function for event-to-event comparison: EMD, GAN
  - [ ] Improve ML-PF model physics performance 
  - [ ] Improve ML-PF model computational performance 
  - [ ] Create CMSSW EDProducer for ML-PF particles
    - [ ] Implement a simple tensorflow or ONNX based ML-PF training for evalutation in CMSSW
- [ ] GPU code for existing PF algorithms
  - [x] test CLUE for element to block clustering
  - [ ] port CLUE to PFBlockAlgo in CMSSW
  - [ ] parallelize PFAlgo calls on blocks
  - [ ] GPU-implementation of PFAlgo
  - [ ] GPU-implementation of PFBlockAlgo distances

## Presentations

- Caltech CMS group ML meeting, 2020-04-16: https://indico.cern.ch/event/909688/contributions/3826957/attachments/2021475/3380133/2020_04_15_caltechml.pdf
- ML4RECO meeting, 2020-04-09: https://indico.cern.ch/event/908361/contributions/3821957/attachments/2017888/3373038/2020_04_08.pdf
- CMS PF group, 2020-03-13: https://indico.cern.ch/event/897397/contributions/3786360/attachments/2003108/3344534/2020_03_13.pdf
- ML4RECO meeting, 2020-03-12: https://indico.cern.ch/event/897281/contributions/3784715/attachments/2002839/3343921/2020_03_12.pdf
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

In case the above links do not load, the presentations are also mirrored on the following CERNBox link: https://cernbox.cern.ch/index.php/s/GkIRJU1YZuai4ix

## Other relevant issues, repos, PR-s:

- https://github.com/jpata/cmssw/issues/56
- https://github.com/cms-sw/cmssw/pull/29361

## Setting up the code

From [setup.sh](test/setup.sh):

```bash
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
cd ../..
```

## Datasets

- April 2020
  - TTbar with PU for PhaseI, privately generated, 20k events 
    - flat ROOT: `/storage/user/jpata/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi/pfntuple_*.root`
    - pickled graph data: `/storage/user/jpata/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi/raw/*.pkl`
    - processed pytorch: `/storage/user/jpata/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi/processed/*.pt`

## Creating the datasets

```bash
cd test
mkdir TTbar_14TeV_TuneCUETP8M1_cfi
python prepare_args.py > args.txt
condor_submit genjob.jdl
```

## Contents of the flat ROOT output ntuple

The ROOT ntuple contains all PFElements, PFCandidates and GenParticles, along with the links. The following code creates the networkx graph data and a normalized data table:

```bash
python test/postprocessing2.py --input data/TTbar_14TeV_TuneCUETP8M1_cfi/pfntuple_1.root --events-per-file 1 --save-full-graph
```

## Contents of the numpy ntuple

For more details, see [data.ipynb](notebooks/data.ipynb). 

## Model training

Train with GenParticles as the target:
```
singularity exec --nv ~jpata/gpuservers/singularity/images/pytorch.simg python3 test/train_end2end.py --dataset data/TTbar_14TeV_TuneCUETP8M1_cfi --n_train 900 --n_test 100 --m
odel PFNet6 --lr 0.001 --hidden_dim 128 --n_epochs 100 --l2 0.001 --l3 10.0 --target gen
```


Train with PFCandidates as the target:
```
singularity exec --nv ~jpata/gpuservers/singularity/images/pytorch.simg python3 test/train_end2end.py --dataset data/TTbar_14TeV_TuneCUETP8M1_cfi --n_train 900 --n_test 100 --m
odel PFNet6 --lr 0.001 --hidden_dim 128 --n_epochs 100 --l2 0.001 --l3 10.0 --target cand
```

## Model validation

Notebook: [test_end2end](notebooks/test_end2end.ipynb)

## Acknowledgements

Part of this work was conducted at **iBanks**, the AI GPU cluster at Caltech. We acknowledge NVIDIA, SuperMicro and the Kavli Foundation for their support of **iBanks**.
