#!/bin/bash
source /opt/hepsim.sh
python3 -m pip install numpy==1.18 networkx==2.4 uproot uproot_methods

cd /opt/hepsim
export HAS_PYTHIA8=true
export PYTHIA8=/opt/hepsim/generators/pythia8
git clone https://github.com/jpata/delphes delphes-local -b angularsmearing
cd delphes-local
./configure
make
