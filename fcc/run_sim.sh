#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out
set -e
set -x

env
df -h

OUTDIR=`pwd`
PFDIR=/home/joosep/particleflow
NEV=100
NUM=$1

#SAMPLE=p8_ee_Z_Ztautau_ecm125
#SAMPLE=p8_ee_tt_ecm365
#SAMPLE=p8_ee_ZZ_fullhad_ecm365
#SAMPLE=p8_ee_qcd_ecm365
#SAMPLE=p8_ee_qcd_ecm380
SAMPLE=p8_ee_ZH_Htautau_ecm380
#SAMPLE=p8_ee_qcd_ecm380
#SAMPLE=p8_ee_gg_ecm365

WORKDIR=/scratch/$USER/${SAMPLE}_${SLURM_JOB_ID}
FULLOUTDIR=${OUTDIR}/${SAMPLE}_overlay365CDR

mkdir -p $FULLOUTDIR

mkdir -p $WORKDIR
cd $WORKDIR

ls -al /cvmfs
ls -al /cvmfs/sw.hsf.org
source /cvmfs/sw.hsf.org/spackages6/key4hep-stack/2022-12-23/x86_64-centos7-gcc11.2.0-opt/ll3gi/setup.sh

cp $PFDIR/fcc/${SAMPLE}.cmd card.cmd
cp $PFDIR/fcc/pythia.py ./
cp $PFDIR/fcc/clic_steer.py ./
cp -R $PFDIR/fcc/PandoraSettings ./
cp -R $PFDIR/fcc/clicRec_e4h_input.py ./

echo "Random:seed=${NUM}" >> card.cmd

k4run $PFDIR/fcc/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd &> log1
ddsim --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml \
      --outputFile out_sim_edm4hep.root \
      --steeringFile clic_steer.py \
      --inputFiles out.hepmc \
      --numberOfEvents $NEV &> log2
k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input out_sim_edm4hep.root --PodioOutput.filename out_reco_edm4hep.root &> log3
cp out_reco_edm4hep.root reco_${SAMPLE}_${NUM}.root

#ddsim --steeringFile clic_steer.py --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml --enableGun --gun.distribution uniform --gun.particle pi- --gun.energy 10*GeV --outputFile piminus_10GeV_edm4hep.root --numberOfEvents $NEV &> log_step1_piminus.txt
#k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input piminus_10GeV_edm4hep.root --PodioOutput.filename piminus_reco.root &> log_step2_piminus.txt

cp reco_${SAMPLE}_${NUM}.root $FULLOUTDIR/

rm -Rf $WORKDIR
