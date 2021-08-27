## Delphes dataset and training

The following instructions use singularity, but you may have a different local setup.

```bash
#Download all pkl.bz2 files from https://zenodo.org/record/4559324

#now move the data into the right place
mv *pythia8_qcd*.pkl.bz2 data/pythia8_qcd/val
mv *pythia8_ttbar*.pkl.bz2 data/pythia8_qcd/raw
mv data/pythia8_qcd/raw/*pythia8_ttbar_9_*.pkl.bz2 data/pythia8_qcd/val

# Generate the TFRecord datasets needed for larger-than-RAM training
python3 mlpf/pipeline.py data -c parameters/delphes.yaml

# Run the training of the base GNN model using e.g. 5 GPUs in a data-parallel mode
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 mlpf/pipeline.py train -c parameters/delphes.yaml

#Run the validation to produce the predictions file
python3 mlpf/pipeline.py evaluate -c parameters/delphes.yaml -t experiments/delphes_* -v "data/pythia8_qcd/val/*.pkl.bz2" -e evaluate_qcd
python3 mlpf/pipeline.py evaluate -c parameters/delphes.yaml -t experiments/delphes_* -v "data/pythia8_ttbar/val/*.pkl.bz2" -e evaluate_ttbar
```

## Recipe for generation
The Delphes AngularSmearing module has been modified to correctly take into account the smearing for tracks, see [delphes/install.sh](delphes/install.sh).

```bash
wget http://atlaswww.hep.anl.gov/hepsim/soft/centos7hepsim.img
sudo singularity build --sandbox centos7hepsim.sandbox centos7hepsim.img
sudo singularity exec -B /home --writable centos7hepsim.sandbox ./install.sh
sudo singularity build centos7hepsim.sif centos7hepsim.sandbox
sudo rm -Rf centos7hepsim.sandbox
```

```bash
cd delphes

# Run the simulation step
# Generate events with pythia, mix them with PU and run a detector simulation using Delphes
singularity exec centos7hepsim.sif ./run_sim.sh

# Run the ntuplization step
# generate X,y input matrices for NN training in out/pythia8_ttbar/*.pkl.bz2
singularity exec centos7hepsim.sif ./run_ntuple.sh
singularity exec centos7hepsim.sif ./run_ntuple_qcd.sh

mv out/pythia8_ttbar ../data/
cd ../data/pythia8_ttbar
mkdir raw
mkdir val
mkdir root
mv *.root root/
mv *.promc root/
mv *.pkl.bz2 raw/
cd ../..

mv out/pythia8_qcd ../data/
cd ../data/pythia8_qcd
mkdir val
mkdir root
mv *.root root/
mv *.promc root/
mv *.pkl.bz2 val/
cd ../..
```
