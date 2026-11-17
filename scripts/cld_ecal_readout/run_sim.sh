#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out
set -e
set -x

env
df -h

export NEV=${NEV:-100}
export SAMPLE=$1 #main card
export NUM=$2 #random seed

#Change these as needed
export OUTDIR=${OUTDIR:-/local/joosep/cld_edm4hep/v1.2.2_key4hep_2025-05-29_CLD_3edac3/}
export CONFIG_DIR=${CONFIG_DIR:-/home/joosep/key4hep-sim/cld/CLDConfig}
export WORKDIR=${WORKDIR:-/scratch/local/$USER/${SAMPLE}_${SLURM_JOB_ID}}
export FULLOUTDIR=${OUTDIR}/${SAMPLE}

mkdir -p "$FULLOUTDIR/root"

mkdir -p $WORKDIR

# Ensure cleanup on exit, even if the job fails
cleanup() {
    if [ ! -z "$WORKDIR" ] && [ "$WORKDIR" != "/scratch/local/$USER" ] && [ "$WORKDIR" != "/scratch/local/$USER/" ]; then
        echo "Cleaning up scratch directory $WORKDIR"
        rm -Rf $WORKDIR
    fi
}
trap cleanup EXIT

cd $WORKDIR

cp -R $CONFIG_DIR ./

cp "${CLD_PYTHIA_CARD:-$CONFIG_DIR/pythia/${SAMPLE}.cmd}" card.cmd

# Add the common event-generator seed for this scan point.
echo "Random:seed=${NUM}" >> card.cmd
cat card.cmd

# Keep geometry selection identical in simulation and reconstruction.
cat > sim.sh <<'EOF'
#!/bin/bash
set -eo pipefail
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2025-05-29
compact_file=${CLD_COMPACT_FILE:-$K4GEO/FCCee/CLD/compact/CLD_o2_v07/CLD_o2_v07.xml}
k4run CLDConfig/pythia.py -n "$NEV" --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
ddsim -I out.hepmc -N -1 -O out_SIM.root --compactFile "$compact_file" --steeringFile CLDConfig/CLDConfig/cld_steer.py
cd CLDConfig/CLDConfig
reco_cms_args=()
if [[ -n "${CLD_CMS_ENERGY:-}" ]]; then
    reco_cms_args=(--cms "$CLD_CMS_ENERGY")
fi
k4run CLDReconstruction.py --inputFiles ../../out_SIM.root --outputBasename out_RECO --num-events -1 --compactFile "$compact_file" "${reco_cms_args[@]}"
EOF

cat sim.sh

#execute the gen-sim-reco step
export PYTHONPATH=$(pwd)/CLDConfig/CLDConfig:$PYTHONPATH
bash sim.sh

ls *.root

#Copy the outputs
cp CLDConfig/CLDConfig/out_RECO_REC.edm4hep.root "$FULLOUTDIR/root/reco_${SAMPLE}_${NUM}.root"
