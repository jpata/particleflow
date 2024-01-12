#!/bin/bash
set -e
set -x

env
df -h

PFDIR=/home/joosep/particleflow
NEV=10

NUM=$1 #random seed
SAMPLE=$2 #main card
NUMPART=$3 #number of particles

OUTDIR=/home/joosep/clic_edm4hep_gun_np$NUMPART/

WORKDIR=/scratch/$USER/${SAMPLE}_${SLURM_JOB_ID}
FULLOUTDIR=${OUTDIR}/${SAMPLE}

mkdir -p $FULLOUTDIR

mkdir -p $WORKDIR
cd $WORKDIR

#cp $PFDIR/fcc/main ./
cp $PFDIR/scripts/fcc/pythia.py ./
cp $PFDIR/scripts/fcc/clic_steer.py ./
cp -R $PFDIR/scripts/fcc/PandoraSettings ./
cp -R $PFDIR/scripts/fcc/clicRec_e4h_input.py ./

#without PU
source /cvmfs/sw.hsf.org/spackages6/key4hep-stack/2023-01-15/x86_64-centos7-gcc11.2.0-opt/csapx/setup.sh

ddsim --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml \
      --steeringFile clic_steer.py \
      --enableGun \
      --gun.distribution uniform \
      --gun.multiplicity $NUMPART \
      --gun.particle $SAMPLE \
      --gun.momentumMin 1*GeV \
      --gun.momentumMax 100*GeV \
      --outputFile out_sim_edm4hep.root \
      --numberOfEvents $NEV \
      --random.seed $NUM
cp out_sim_edm4hep.root $FULLOUTDIR/sim_${SAMPLE}_${NUM}.root

\time -v k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input out_sim_edm4hep.root --PodioOutput.filename out_reco_edm4hep.root
cp out_reco_edm4hep.root $FULLOUTDIR/reco_${SAMPLE}_${NUM}.root
cp timing_histos.root $FULLOUTDIR/timing_${SAMPLE}_${NUM}.root

rm -Rf $WORKDIR
