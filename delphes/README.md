## Delphes dataset and training

The following instructions use singularity, but you may have a different local setup.

```bash
cd delphes

# Run the simulation step
# Generate events with pythia, mix them with PU and run a detector simulation using Delphes
singularity exec http://jpata.web.cern.ch/jpata/centos7hepsim.sif ./run_sim.sh

# Run the ntuplization step
# generate X,y input matrices for NN training in out/pythia8_ttbar/*.pkl.bz2
singularity exec http://jpata.web.cern.ch/jpata/centos7hepsim.sif ./run_ntuple.sh

#Alternatively, to skip run_sim.sh and run_ntuple.sh, download everything from https://doi.org/10.5281/zenodo.4452283 and put into out/pythia8_ttbar

# Generate the TFRecord datasets needed for larger-than-RAM training
singularity exec --nv http://jpata.web.cern.ch/jpata/base.simg python3 ../mlpf/tensorflow/delphes_data.py --datapath out/pythia8_ttbar

# Run the training of the base GNN model using e.g. 5 GPUs in a data-parallel mode
CUDA_VISIBLE_DEVICES=0,1,2,3,4 singularity exec --nv http://jpata.web.cern.ch/jpata/base.simg python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/delphes-gnn-skipconn.yaml --action train

#Run the validation to produce the predictions file
singularity exec --nv http://jpata.web.cern.ch/jpata/base.simg python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/delphes-gnn-skipconn.yaml --action validate --weights ./experiments/delphes-gnn-skipconn-*/weights.300-*.hdf5

singularity exec --nv http://jpata.web.cern.ch/jpata/base.simg python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/delphes-gnn-skipconn.yaml --action timing --weights ./experiments/delphes-gnn-skipconn-*/weights.300-*.hdf5
```

## Recipe to prepare Delphes singularity image
NB: The Delphes AngularSmearing module has been modified to take into account the smearing, see [install.sh](install.sh)

```bash
wget http://atlaswww.hep.anl.gov/hepsim/soft/centos7hepsim.img
sudo singularity build --sandbox centos7hepsim.sandbox centos7hepsim.img
sudo singularity exec -B /home --writable centos7hepsim.sandbox ./install.sh
sudo singularity build centos7hepsim.sif centos7hepsim.sandbox
sudo rm -Rf centos7hepsim.sandbox
```