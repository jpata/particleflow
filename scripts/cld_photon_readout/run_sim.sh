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
export GUN_PARTICLE=${GUN_PARTICLE:-}
export GUN_ENERGY=${GUN_ENERGY:-10*GeV}
export GUN_DISTRIBUTION=${GUN_DISTRIBUTION:-uniform}

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

if [[ "${SAMPLE}" == gun_* ]]; then
    if [[ -z "${GUN_PARTICLE}" ]]; then
        case "${SAMPLE}" in
            gun_e_*) GUN_PARTICLE=e- ;;
            gun_gamma_*) GUN_PARTICLE=gamma ;;
            gun_mu_*) GUN_PARTICLE=mu- ;;
            gun_pi_*) GUN_PARTICLE=pi- ;;
            *)
                echo "Unsupported CLD particle-gun sample: ${SAMPLE}" >&2
                exit 1
                ;;
        esac
    fi
else
    cp "${CLD_PYTHIA_CARD:-$CONFIG_DIR/pythia/${SAMPLE}.cmd}" card.cmd

    #add seed to pythia card
    echo "Random:seed=${NUM}" >> card.cmd
    cat card.cmd
fi

# Keep geometry selection identical in simulation and reconstruction.
cat > sim.sh <<'EOF'
#!/bin/bash
set -eo pipefail
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2025-05-29
compact_file=${CLD_COMPACT_FILE:-$K4GEO/FCCee/CLD/compact/CLD_o2_v07/CLD_o2_v07.xml}
if [[ "$SAMPLE" == gun_* ]]; then
    ddsim --enableGun --gun.distribution "$GUN_DISTRIBUTION" --gun.energy "$GUN_ENERGY" --gun.particle "$GUN_PARTICLE" --crossingAngleBoost 0.0 --random.enableEventSeed --random.seed "$NUM" -N "$NEV" -O out_SIM.root --compactFile "$compact_file" --steeringFile CLDConfig/CLDConfig/cld_steer.py
else
    k4run CLDConfig/pythia.py -n "$NEV" --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
    ddsim -I out.hepmc -N -1 -O out_SIM.root --compactFile "$compact_file" --steeringFile CLDConfig/CLDConfig/cld_steer.py
fi
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
if [[ "${SAVE_SIM:-0}" == "1" ]]; then
    cp out_SIM.root "$FULLOUTDIR/root/sim_${SAMPLE}_${NUM}.root"
fi
