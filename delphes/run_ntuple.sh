#!/bin/bash

source /opt/hepsim.sh
export LD_LIBRARY_PATH=/opt/hepsim/delphes:$LD_LIBRARY_PATH
export ROOT_INCLUDE_PATH=/opt/hepsim/delphes:/opt/hepsim/delphes/external

XDIR="out/pythia8_ttbar"
mkdir -p $XDIR
#rm -f $XDIR/*.pkl

for NUM in `seq 0 9`; do
  INROOT="tev14_pythia8_ttbar_$NUM.root"
  OUTPKL="tev14_pythia8_ttbar_$NUM.pkl.bz2"
  python ntuplizer.py $XDIR/$INROOT $XDIR/$OUTPKL
done
