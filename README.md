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
