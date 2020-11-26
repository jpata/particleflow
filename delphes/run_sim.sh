# a simple test
# run over 14 TeV events
# wget http://atlaswww.hep.anl.gov/hepsim/soft/centos7hepsim.img
# singularity exec centos7hepsim.img run_sim.sh

#XFILE="tev14_pythia8_qcdjets_wgt_001"
#
#if [ ! -f ${XFILE}.promc ]; then
#    wget http://mc.hep.anl.gov/asc/hepsim/events/pp/14tev/pythia8_qcdjets_wgt/${XFILE}.promc
#fi

source /opt/hepsim.sh
make -f Makefile

XDIR="out/pythia8_ttbar"
mkdir -p $XDIR 
rm -f $XDIR/*.promc $XDIR/*.root


NSYST=1
echo "Run  $NSYST jobs and collect files in $XDIR/"

n=0
#------------------------------ start loop ----------------------------
while  [ $n -lt $NSYST ]
do
  echo "------------ Do run = $n" 
  NUM=`printf "%03d" $n`
  OUT="tev14_pythia8_ttbar_$NUM.promc"
  OUTROOT="tev14_pythia8_ttbar_$NUM.root"
  LOG="logfile_$NUM.txt"
  echo "running.."
  ./main.exe tev14_pythia8_ttbar.py pythia8.promc > $XDIR/$LOG 2>&1
  echo "file  $OUT done for run $n "
  mv pythia8.promc  $XDIR/$OUT
  DelphesProMC delphes_card_CMS_PileUp.tcl $XDIR/$OUTROOT $XDIR/$OUT >> $XDIR/$LOG 2>&1
  let "n = $n + 1"
done
