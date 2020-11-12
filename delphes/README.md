Generating Delphes events with pileup and converting them to flat uproot-capable files.

```bash
# Run the simulation step
singularity exec http://jpata.web.cern.ch/jpata/centos7hepsim.sif ./run_sim.sh

# Run the ntuplization step
singularity exec http://jpata.web.cern.ch/jpata/centos7hepsim.sif ./run_ntuple.sh

# Run the training
singularity exec --nv http://jpata.web.cern.ch/jpata/base.simg python3 ../mlpf/tensorflow/delphes_model.py

#recipe to prepare singularity image from scratch
# wget http://atlaswww.hep.anl.gov/hepsim/soft/centos7hepsim.img
# sudo singularity exec -B /home --writable centos7hepsim.img ./install.sh
# sudo singularity build centos7hepsim.sif centos7hepsim.img

```
