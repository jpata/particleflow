
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

Running the code
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
