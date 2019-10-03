
Setting up the code
```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
scramv1 project CMSSW CMSSW_11_0_0_pre7
cd CMSSW_11_0_0_pre7/src
eval `scramv1 runtime -sh`
git cms-init
mkdir workspace
git clone https://github.com/jpata/particleflow.git workspace/particleflow 

```

Small standalone example
```bash
cmsRun test/step3.py
python3 test/ntuplizer.py ./data step3_AOD.root
root -l ./data/step3_AOD.root
```

Running the code on larger samples
```bash
#Run the crab jobs
cd test
python multicrab.py
cd ..

#Make the ntuple
./test/run_ntuple.sh /path/to/crab/output ./data/DATASET

#Run the tensorflow training
./test/run_training.sh ./data/DATASET
```

PFCandidate <-> element associations
```
root [4] pftree->Scan("pfcands_nelem:pfcands_iblock:pfcands_ielem0:pfcands_ielem1")
***********************************************************************
*    Row   * Instance * pfcands_n * pfcands_i * pfcands_i * pfcands_i *
***********************************************************************
*        0 *        0 *         2 *       146 *        33 *        39 *
*        0 *        1 *         5 *         1 *        74 *       261 *
*        0 *        2 *         2 *       146 *        34 *        39 *
*        0 *        3 *        11 *         1 *        42 *       472 *

root [1] pftree->Scan("clusters_npfcands:clusters_ipfcand0:clusters_ipfcand1")
***********************************************************
*    Row   * Instance * clusters_ * clusters_ * clusters_ *
***********************************************************
*        0 *        0 *         3 *       381 *       827 *
*        0 *        1 *         1 *      2125 *         0 *
*        0 *        2 *         1 *       814 *         0 *
*        0 *        3 *         1 *       449 *         0 *
*        0 *        4 *         3 *       365 *       626 *
*        0 *        5 *         3 *       625 *       637 *
*        0 *        6 *         2 *       120 *       484 *
*        0 *        7 *         2 *       965 *      1057 *
*        0 *        8 *         6 *       307 *       470 *

root [2] pftree->Scan("tracks_npfcands:tracks_ipfcand0:tracks_ipfcand1")
***********************************************************
*    Row   * Instance * tracks_np * tracks_ip * tracks_ip *
***********************************************************
*        0 *        0 *         1 *      1836 *         0 *
...
*        0 *        1 *         0 *         0 *         0 *
*        0 *       16 *         0 *         0 *         0 *
*        0 *       17 *         0 *         0 *         0 *
*        0 *       18 *         1 *      1458 *         0 *
*        0 *       19 *         2 *       401 *      1367 *
*        0 *       20 *         1 *       134 *         0 *
*        0 *       21 *         2 *       191 *      1401 *
*        0 *       22 *         1 *       985 *         0 *

```
