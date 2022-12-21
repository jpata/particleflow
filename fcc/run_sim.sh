#!/bin/bash
set -e

#NEV=5000
#ddsim --steeringFile clic_steer.py --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml --enableGun --gun.distribution uniform --gun.particle pi- --gun.energy 10*GeV --outputFile piminus_10GeV_edm4hep.root --numberOfEvents $NEV &> log_step1_piminus.txt
#k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input piminus_10GeV_edm4hep.root --PodioOutput.filename piminus_reco.root &> log_step2_piminus.txt

NEV=100
#samplename=p8_ee_Z_Ztautau_ecm125
#samplename=p8_ee_tt_ecm365
samplename=p8_ee_ZZ_fullhad_ecm365
k4run pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard ${samplename}.cmd &> log_step1.txt
ddsim --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml \
      --outputFile out_sim_edm4hep.root \
      --steeringFile clic_steer.py \
      --inputFiles out.hepmc \
      --numberOfEvents $NEV &> log_step2.txt
k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input out_sim_edm4hep.root --PodioOutput.filename out_reco_edm4hep.root &> log_step3.txt
cp out_reco_edm4hep.root reco_${samplename}.root
