```
#Initialize centos 7
cmssw-el7

export SCRAM_ARCH=slc7_amd64_gcc10
cmsrel CMSSW_12_3_0_pre6
cd CMSSW_12_3_0_pre6
cmsenv
git cms-init

#checkout the MLPF code
git-cms-merge-topic jpata:pfanalysis_caloparticle

#check out the version from the 2022 release
git checkout 547a0fce7251bfaa6e855aef068f5a45c2d321ec

#compile
scram b -j4
```
