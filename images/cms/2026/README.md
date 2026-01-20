```bash
#build the alma8 image with rendering libraries
apptainer build alma8.simg alma8.singularity
apptainer exec -B /cvmfs alma8.simg /bin/bash

#initialize CMSSW
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_15_0_5
cd CMSSW_15_0_5
cmsenv
cd ..
cmsShow -i CMSSW_15_0_5_mlpf_v2.6.0pre1_puppi_2372e2_small/cuda_False/JetMET0_mlpf/step3_RECO_1.root -c images/cms/2026/fireworks_reco.fwc
```
