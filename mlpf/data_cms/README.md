```
export IMG=/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel7
export SCRAM_ARCH=slc7_amd64_gcc10

cmsrel CMSSW_12_3_0_pre6
cd CMSSW_12_3_0_pre6
git cms-init
git-cms-merge-topic jpata:pfanalysis_caloparticle
scram b -j4
```
