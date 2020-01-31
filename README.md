Notes on modernizing CMS particle flow, in particular [PFBlockAlgo](https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/src/PFBlockAlgo.cc) and [PFAlgo](https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/src/PFAlgo.cc).

# Overview

- [x] set up datasets and ntuples for detailed PF analysis
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
  - [ ] set up datasets for regression genparticles from elements
    - [ ] develop improved loss function for event-to-event comparison: EMD, GAN?
## Presentations

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
```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820
scramv1 project CMSSW CMSSW_11_0_0_pre12
cd CMSSW_11_0_0_pre12/src
eval `scramv1 runtime -sh`
git cms-init
mkdir workspace
git clone https://github.com/jpata/particleflow.git workspace/particleflow 
```


## Running the RECO step with particle flow
```
#Run 3
cmsDriver.py step3  --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO --runUnscheduled  --conditions auto:phase1_2021_realistic -s RAW2DIGI,L1Reco,RECO,RECOSIM,EI,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM --eventcontent RECOSIM,MINIAODSIM,DQM -n 100  --filein  file:step2.root  --fileout file:step3.root --no_exec --era Run3 --scenario pp --geometry DB:Extended --mc
```

## Small standalone example
```bash
cmsRun test/step3.py

#Produce the flat root ntuple
python3 test/ntuplizer.py ./data step3_AOD.root
root -l ./data/step3_AOD.root

#Produce the numpy datasets
python3 test/graph.py ./data/step3_AOD.root
ls ./data/step3_AOD_*.npz

```

## Running on grid
```bash
#Run the crab jobs
cd test
python multicrab.py
cd ..

#Make the ROOT ntuple (edit Makefile first)
make ntuples

#make the numpy cache
make cache
```

## Datasets

- November 27, 2019
  - /RelValTTbar_14TeV/CMSSW_11_0_0_pre12-PU_110X_mcRun3_2021_realistic_v5-v1/GEN-SIM-DIGI-RAW
    - EDM: /mnt/hadoop/store/user/jpata/RelValTTbar_14TeV/pfvalidation/191126_233751/0000/step3_AOD*.root
    - flat ROOT: /storage/user/jpata/particleflow/data/TTbar_run3/step3_ntuple_*.root or /eos/user/j/jpata/particleflow/TTbar_run3/step3_AOD*.root
    - npy: /storage/user/jpata/particleflow/data/TTbar_run3/step3_ntuple_*.npz
  - /RelValQCD_FlatPt_15_3000HS_14/CMSSW_11_0_0_pre12-PU_110X_mcRun3_2021_realistic_v5-v1/GEN-SIM-DIGI-RAW
    - EDM: /mnt/hadoop/store/user/jpata/RelValQCD_FlatPt_15_3000HS_14/pfvalidation/191126_233511/0000/step3_AOD*.root 
    - flat ROOT: /storage/user/jpata/particleflow/data/QCD_run3/step3_ntuple_*.root or /eos/user/j/jpata/particleflow/QCD_run3/step3_AOD*.root
    - npy: /storage/user/jpata/particleflow/data/QCD_run3/step3_ntuple_*.npz
  - /RelValNuGun/CMSSW_11_0_0_pre12-PU_110X_mcRun3_2021_realistic_v5-v1/GEN-SIM-DIGI-RAW
    - EDM: /mnt/hadoop/store/user/jpata/RelValNuGun/pfvalidation/191126_233630/0000/step3_AOD*.root
    - flat ROOT: /storage/user/jpata/particleflow/data/NuGun_run3/step3_ntuple_*.root or /eos/user/j/jpata/particleflow/NuGun_run3/step3_AOD*.root
    - npy: /storage/user/jpata/particleflow/data/NuGun_run3/step3_ntuple_*.npz

- October 9, 2019
  - /RelValTTbar_13/CMSSW_11_0_0_pre6-PU25ns_110X_upgrade2018_realistic_v3-v1/GEN-SIM-DIGI-RAW
  - size: 9000 events
  - code version: 712e6d6
  - EDM: /mnt/hadoop/store/user/jpata/RelValTTbar_13/pfvalidation/191004_163947/0000/step3_AOD*.root
  - flat ROOT: /storage/user/jpata/particleflow/data/TTbar/191009_155100/step3_AOD_*.root or /eos/user/j/jpata/particleflow/TTbar/191009_155100/step3_AOD_*.root
  - npy: /storage/user/jpata/particleflow/data/TTbar/191009_155100/step3_AOD_*.npz 

## Contents of the flat ROOT output ntuple

The TTree `pftree` contains the elements, candidates and genparticles:
- clusters ([PFRecCluster](https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowReco/interface/PFCluster.h))
  - `PFBlockElement::(ECAL, PS1, PS2, HCAL, GSF, HO, HFHAD, HFEM)`           
- tracks ([PFRecTrack](https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowReco/interface/PFRecTrack.h))
  - `PFBlockElement::TRACK` or `PFBlockElement::BREM`
- candidates ([PFCandidates](https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/interface/PFCandidate.h))
- genparticles

```
pftree->Print()
*Br    0 :nclusters : nclusters/i                                            *
*Br    1 :clusters_iblock : clusters_iblock[nclusters]/i                     *
*Br    2 :clusters_ielem : clusters_ielem[nclusters]/i                       *
*Br    3 :clusters_ipfcand0 : clusters_ipfcand0[nclusters]/i                 *
*Br    4 :clusters_ipfcand1 : clusters_ipfcand1[nclusters]/i                 *
*Br    5 :clusters_ipfcand2 : clusters_ipfcand2[nclusters]/i                 *
*Br    6 :clusters_ipfcand3 : clusters_ipfcand3[nclusters]/i                 *
*Br    7 :clusters_npfcands : clusters_npfcands[nclusters]/i                 *
*Br    8 :clusters_layer : clusters_layer[nclusters]/I                       *
*Br    9 :clusters_depth : clusters_depth[nclusters]/I                       *
*Br   10 :clusters_type : clusters_type[nclusters]/I                         *
*Br   11 :clusters_energy : clusters_energy[nclusters]/F                     *
*Br   12 :clusters_x : clusters_x[nclusters]/F                               *
*Br   13 :clusters_y : clusters_y[nclusters]/F                               *
*Br   14 :clusters_z : clusters_z[nclusters]/F                               *
*Br   15 :clusters_eta : clusters_eta[nclusters]/F                           *
*Br   16 :clusters_phi : clusters_phi[nclusters]/F                           *
*Br   17 :ngenparticles : ngenparticles/i                                    *
*Br   18 :genparticles_pt : genparticles_pt[ngenparticles]/F                 *
*Br   19 :genparticles_eta : genparticles_eta[ngenparticles]/F               *
*Br   20 :genparticles_phi : genparticles_phi[ngenparticles]/F               *
*Br   21 :genparticles_x : genparticles_x[ngenparticles]/F                   *
*Br   22 :genparticles_y : genparticles_y[ngenparticles]/F                   *
*Br   23 :genparticles_z : genparticles_z[ngenparticles]/F                   *
*Br   24 :genparticles_pdgid : genparticles_pdgid[ngenparticles]/I           *
*Br   25 :ntracks   : ntracks/i                                              *
*Br   26 :tracks_iblock : tracks_iblock[ntracks]/i                           *
*Br   27 :tracks_ielem : tracks_ielem[ntracks]/i                             *
*Br   28 :tracks_ipfcand0 : tracks_ipfcand0[ntracks]/i                       *
*Br   29 :tracks_ipfcand1 : tracks_ipfcand1[ntracks]/i                       *
*Br   30 :tracks_ipfcand2 : tracks_ipfcand2[ntracks]/i                       *
*Br   31 :tracks_ipfcand3 : tracks_ipfcand3[ntracks]/i                       *
*Br   32 :tracks_npfcands : tracks_npfcands[ntracks]/i                       *
*Br   33 :tracks_qoverp : tracks_qoverp[ntracks]/F                           *
*Br   34 :tracks_lambda : tracks_lambda[ntracks]/F                           *
*Br   35 :tracks_phi : tracks_phi[ntracks]/F                                 *
*Br   36 :tracks_eta : tracks_eta[ntracks]/F                                 *
*Br   37 :tracks_dxy : tracks_dxy[ntracks]/F                                 *
*Br   38 :tracks_dsz : tracks_dsz[ntracks]/F                                 *
*Br   39 :tracks_outer_eta : tracks_outer_eta[ntracks]/F                     *
*Br   40 :tracks_outer_phi : tracks_outer_phi[ntracks]/F                     *
*Br   41 :tracks_inner_eta : tracks_inner_eta[ntracks]/F                     *
*Br   42 :tracks_inner_phi : tracks_inner_phi[ntracks]/F                     *
*Br   43 :npfcands  : npfcands/i                                             *
*Br   44 :pfcands_pt : pfcands_pt[npfcands]/F                                *
*Br   45 :pfcands_eta : pfcands_eta[npfcands]/F                              *
*Br   46 :pfcands_phi : pfcands_phi[npfcands]/F                              *
*Br   47 :pfcands_charge : pfcands_charge[npfcands]/F                        *
*Br   48 :pfcands_energy : pfcands_energy[npfcands]/F                        *
*Br   49 :pfcands_pdgid : pfcands_pdgid[npfcands]/I                          *
*Br   50 :pfcands_nelem : pfcands_nelem[npfcands]/I                          *
*Br   51 :pfcands_ielem0 : pfcands_ielem0[npfcands]/I                        *
*Br   52 :pfcands_ielem1 : pfcands_ielem0[npfcands]/I                        *
*Br   53 :pfcands_ielem2 : pfcands_ielem0[npfcands]/I                        *
*Br   54 :pfcands_ielem3 : pfcands_ielem0[npfcands]/I                        *
*Br   55 :pfcands_iblock : pfcands_iblock[npfcands]/I                        *
```

Distance matrix between elements:
```
root [2] linktree->Print()
*Tree    :linktree  : linktree for elements in block                         *
*Br    0 :nlinkdata : nlinkdata/i                                            *
*Br    1 :linkdata_distance : linkdata_distance[nlinkdata]/F                 *
*Br    2 :linkdata_iev : linkdata_iev[nlinkdata]/i                           *
*Br    3 :linkdata_iblock : linkdata_iblock[nlinkdata]/i                     *
*Br    4 :linkdata_ielem : linkdata_ielem[nlinkdata]/i                       *
*Br    5 :linkdata_jelem : linkdata_jelem[nlinkdata]/i                       *
```

Element to candidate association:
```
root [2] linkdata_elemtocand->Print()
*Br    0 :nlinkdata_elemtocand : nlinkdata_elemtocand/i                      *
*Br    1 :linkdata_elemtocand_iev :                                          *
*Br    2 :linkdata_elemtocand_iblock :                                       *
*Br    3 :linkdata_elemtocand_ielem :                                        *
*Br    4 :linkdata_elemtocand_icand :                                        *
```
## Numpy training ntuples

Produced using

```bash
python3 test/graph.py step3_AOD_1.root
```
- step3_AOD_1_ev.npz: PF elements, candidates, and the block associations via a numerical ID
  - elements: [Nelem, Nelem_feat] for the input PFElement data
  - element_block_id: [Nelem, ] for the PFAlgo-based block id
  - candidates: [Ncand, Ncand_feat] for the output PFCandidate data
  - candidate_block_id: [Ncand, ] for the PFAlgo-based block id 
- step3_AOD_1_dist.npz: sparse [Nelem, Nelem] distance matrix from PFBlockAlgo between the candidates

## Standard CMS offline PF
The following pseudocode illustrates how the standard offline PF works in CMS.

```python
# Inputs and outputs of Particle Flow
# elements: array of ECAL cluster, HCAL cluster, tracks etc, size Nelem
# candidates: array of produced particle flow candidates (pions, kaons, photons etc)

# Intermediate data structures
# link_matrix: whether or not two elements are linked by having a finite distance (sparse, Nelem x Nelem)

def particle_flow(elements):

    #based on https://github.com/cms-sw/cmssw/tree/master/RecoParticleFlow/PFProducer/plugins/linkers
    link_matrix = compute_links(elements)
    
    #based on https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/src/PFBlockAlgo.cc
    blocks = create_blocks(elements, link_matrix)
    
    #based on https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/src/PFAlgo.cc
    candidates = []
    for block in blocks:
        candidates.append(create_candidates(block))
    
    return candidates
    
def compute_links(elements):
    Nelem = len(elements)
 
    link_matrix = np.array((Nelem, Nelem))
    link_matrix[:] = 0
    
    #test if two elements are close by based on neighborhood implemented with KD-trees
    for ielem in range(Nelem):
        for jelem in range(Nelem):
            if in_neighbourhood(elements, ielem, jelem):
                link_matrix[ielem, jelem] = 1
                
    return link_matrix

def in_neighbourhood(elements, ielem, jelem):
    #This element-to-element neighborhood checking is done based on detector geometry
    #e.g. here for TRK to ECAL: https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/plugins/linkers/TrackAndECALLinker.cc -> linkPrefilter
    return True

def distance(elements, ielem, jelem):
    #This element-to-element distance checking is done based on detector geometry
    #e.g. here for TRK to ECAL: https://github.com/cms-sw/cmssw/blob/master/RecoParticleFlow/PFProducer/plugins/linkers/TrackAndECALLinker.cc -> testLink
    return 0.0
    
def create_blocks(elements, link_matrix):
    #Each block is a list of elements, this is a list of blocks
    blocks = []
    
    Nelem = len(elements)

    #Elements and connections between the elements
    graph = Graph()
    for ielem in range(Nelem):
        graph.add_node(ielem)
    
    #Check the distance between all relevant element pairs
    for ielem in range(Nelem):
        for jelem in range(Nelem):
            if link_matrix[ielem, jelem]:
                dist = distance(elements, ielem, jelem)
                if dist > -0.5:
                    graph.add_edge(ielem, jelem)
    
    #Find the sets of elements that are connected
    for subgraph in find_subgraphs(graph):
        this_block = []
        for element in subgraph:
            this_block.append(element)
        blocks.append(this_block)

    return blocks   

def create_candidates(block):
    #find all HCAL-ECAL-TRK triplets, produce pions
    #find all HCAL-TRK pairs, produce kaons
    #find all ECAL-TRK pairs, produce pions
    #find all independent ECAL elements, produce photons
    #etc etc
    candidates = []
    return candidates
```


## Acknowledgements

Part of this work was conducted at **iBanks**, the AI GPU cluster at Caltech. We acknowledge NVIDIA, SuperMicro and the Kavli Foundation for their support of **iBanks**.
