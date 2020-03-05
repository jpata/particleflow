Generating Delphes events with pileup and converting them to flat uproot-capable files.

```bash
#Download the singularity image
wget http://atlaswww.hep.anl.gov/hepsim/soft/centos7hepsim.img

# Install numpy inside the image
sudo singularity exec --writable centos7hepsim.img python -m pip install numpy
  
# Run the simulation step
singularity exec centos7hepsim.img run_sim.sh

# Run the ntuplization step
singularity exec centos7hepsim.img run_ntuple.sh
```
