#!/bin/bash
set +e

NUM=$1

XDIR="out/pythia8_qcd"
mkdir -p $XDIR
OUTROOT="tev14_pythia8_qcd_$NUM.root"
OUT="tev14_pythia8_qcd_$NUM.promc"
LOG="logfile_$NUM.txt"

rm -f $XDIR/$OUTROOT $XDIR/$OUT

source /opt/hepsim.sh
cp tev14_pythia8_qcd.py tev14_pythia8_qcd.py.${NUM}
echo "Random:seed=${NUM}" >> tev14_pythia8_.py.${NUM}
./main.exe tev14_pythia8_qcd.py.${NUM} $XDIR/$OUT > $XDIR/$LOG 2>&1
/opt/hepsim/delphes-local/DelphesProMC delphes_card_CMS_PileUp.tcl $XDIR/$OUTROOT $XDIR/$OUT >> $XDIR/$LOG 2>&1
