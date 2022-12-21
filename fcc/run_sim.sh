#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out
set -e
set -x

env
df -h

WORKDIR=/scratch/$USER/${SLURM_JOB_ID}
SAMPLE=$1
SEED=$2
OUTDIR=/local/joosep/mlpf/gen/clic/
PFDIR=/home/joosep/particleflow
NEV=20000
NUM=1

#SAMPLE=p8_ee_Z_Ztautau_ecm125
SAMPLE=p8_ee_tt_ecm365
#SAMPLE=p8_ee_ZZ_fullhad_ecm365

mkdir -p $WORKDIR
cd $WORKDIR

ls -al /cvmfs
ls -al /cvmfs/sw.hsf.org
source /cvmfs/sw.hsf.org/key4hep/setup.sh

cp $PFDIR/fcc/${SAMPLE}.cmd card.cmd
cp $PFDIR/fcc/pythia.py ./
cp $PFDIR/fcc/clic_steer.py ./
cp -R $PFDIR/fcc/PandoraSettings ./
cp -R $PFDIR/fcc/clicRec_e4h_input.py ./

echo "Random:seed=${NUM}" >> card.cmd

k4run $PFDIR/fcc/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
ddsim --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml \
      --outputFile out_sim_edm4hep.root \
      --steeringFile clic_steer.py \
      --inputFiles out.hepmc \
      --numberOfEvents $NEV
k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input out_sim_edm4hep.root --PodioOutput.filename out_reco_edm4hep.root
cp out_reco_edm4hep.root reco_${SAMPLE}.root

#ddsim --steeringFile clic_steer.py --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml --enableGun --gun.distribution uniform --gun.particle pi- --gun.energy 10*GeV --outputFile piminus_10GeV_edm4hep.root --numberOfEvents $NEV &> log_step1_piminus.txt
#k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input piminus_10GeV_edm4hep.root --PodioOutput.filename piminus_reco.root &> log_step2_piminus.txt

cp reco_${SAMPLE}.root $OUTDIR/$SAMPLE/

rm -Rf $WORKDIR
