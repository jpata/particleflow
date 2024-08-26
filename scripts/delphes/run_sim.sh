#!/bin/bash

set +e

source /opt/hepsim.sh
make -f Makefile

XDIR="out/pythia8_ttbar"
mkdir -p $XDIR

./run_pileup.sh

for i in `seq 0 9`; do
  nohup ./run_sim_seed.sh $i &
done
nohup ./run_sim_seed_qcd.sh 10 &

wait
