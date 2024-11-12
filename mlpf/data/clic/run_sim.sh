#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out
set -e
set -x

env
df -h

OUTDIR=/local/joosep/clic_edm4hep_2023_02_27/
PFDIR=/home/joosep/particleflow
NEV=100

NUM=$1 #random seed
SAMPLE=$2 #main card
#PU=$3 #pu card

WORKDIR=/scratch/local/$USER/${SAMPLE}_${SLURM_JOB_ID}
FULLOUTDIR=${OUTDIR}/${SAMPLE}

mkdir -p $FULLOUTDIR

mkdir -p $WORKDIR
cd $WORKDIR

#cp $PFDIR/fcc/main ./
cp $PFDIR/fcc/${SAMPLE}.cmd card.cmd
#cp $PFDIR/fcc/${PU}.cmd card_pu.cmd
cp $PFDIR/fcc/pythia.py ./
cp $PFDIR/fcc/clic_steer.py ./
cp -R $PFDIR/fcc/PandoraSettings ./
cp -R $PFDIR/fcc/clicRec_e4h_input.py ./

echo "Random:seed=${NUM}" >> card.cmd
cat card.cmd

#without PU
source /cvmfs/sw.hsf.org/spackages6/key4hep-stack/2023-01-15/x86_64-centos7-gcc11.2.0-opt/csapx/setup.sh
k4run $PFDIR/fcc/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd

ddsim --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml \
      --steeringFile clic_steer.py \
      --outputFile out_sim_edm4hep.root \
      --inputFiles out.hepmc \
      --numberOfEvents $NEV \
      --random.seed $NUM

k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input out_sim_edm4hep.root --PodioOutput.filename out_reco_edm4hep.root
cp out_reco_edm4hep.root reco_${SAMPLE}_${NUM}.root

cp reco_${SAMPLE}_${NUM}.root $FULLOUTDIR/

rm -Rf $WORKDIR
