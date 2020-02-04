# a simple test
# run over 14 TeV events
# wget http://atlaswww.hep.anl.gov/hepsim/soft/centos7hepsim.img
# singularity exec centos7hepsim.img run_sim.sh

XFILE="tev14_pythia8_qcdjets_wgt_001"

if [! -f ${XFILE}.promc ]; then
    wget http://mc.hep.anl.gov/asc/hepsim/events/pp/14tev/pythia8_qcdjets_wgt/${XFILE}.promc
fi

source /opt/hepsim.sh

rm -f out.root
DelphesProMC delphes_card_CMS_PileUp.tcl out.root ${XFILE}.promc
