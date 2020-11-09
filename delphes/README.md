Generating Delphes events with pileup and converting them to flat uproot-capable files.

```bash
#Download the singularity image
wget http://atlaswww.hep.anl.gov/hepsim/soft/centos7hepsim.img

# Install numpy & networkx inside the image
sudo singularity exec -B /home --writable centos7hepsim.img ./install.sh
  
# Run the simulation step
singularity exec centos7hepsim.img ./run_sim.sh

# Run the ntuplization step
singularity exec centos7hepsim.img ./run_ntuple.sh
```
