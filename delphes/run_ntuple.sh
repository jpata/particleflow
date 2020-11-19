#!/bin/bash

source /opt/hepsim.sh
export LD_LIBRARY_PATH=/opt/hepsim/delphes:$LD_LIBRARY_PATH
export ROOT_INCLUDE_PATH=/opt/hepsim/delphes:/opt/hepsim/delphes/external

XDIR="out/pythia8_ttbar"
mkdir -p $XDIR
rm -f $XDIR/*.pkl

NSYST=1
echo "Run  $NSYST jobs and collect files in $XDIR/"

n=0
#------------------------------ start loop ----------------------------                                                           
while  [ $n -lt $NSYST ]
do
  echo "------------ Do run = $n"
  NUM=`printf "%03d" $n`
  INROOT="tev14_pythia8_ttbar_$NUM.root"
  OUTPKL="tev14_pythia8_ttbar_$NUM.pkl"
  python ntuplizer.py $XDIR/$INROOT $XDIR/$OUTPKL
  let "n = $n + 1"
done
