#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source_dir=$repo_dir/CMSSW_20_1_0_pre5/34506.213_SinglePiFlatPt0p7To10+Run4D121_HGCALNanoVal
mkdir -p "$repo_dir/logs/cmssw"
run_dir=$(mktemp -d "$repo_dir/logs/cmssw/$(date -u +%Y%m%dT%H%M%SZ)-mixedguns.XXXXXX")
cp "$source_dir/SinglePiFlatPt0p7To10_cfi_GEN_SIM.py" "$source_dir/step"{2,3}_*.py "$run_dir/"
cat >> "$run_dir/SinglePiFlatPt0p7To10_cfi_GEN_SIM.py" <<'PY'
# Ten events, each containing a particle/antiparticle pair of every species.
process.generator.PGunParameters.ParticleID = cms.vint32(11, 13, 22, 130, 211)
process.generator.psethack = cms.string('mixed electron muon photon K0L pion gun, pt 0.7-10 GeV')
PY
printf '%s\n' "$run_dir" > "$repo_dir/logs/cmssw/latest_mixed_gun_test.txt"
export APPTAINER_NV=true
export SITECONFIG_PATH=/cvmfs/cms.cern.ch/SITECONF/T2_EE_Estonia
export CMSSW_RUN_DIR=$run_dir
printf 'Run directory: %s\n' "$run_dir"
"$repo_dir/env.sh" bash -c '
set -euo pipefail
ulimit -c 0
configs=(SinglePiFlatPt0p7To10_cfi_GEN_SIM.py step2_*.py step3_*.py)
for i in 0 1 2; do
  step=$((i+1))
  set +e
  cmsRun -j "step${step}_FrameworkJobReport.xml" "$PWD/${configs[i]}" 2>&1 | tee "step${step}_run.log"
  status=${PIPESTATUS[0]}
  set -e
  printf "%s\n" "$status" > "step${step}_exit_code"
  if [[ -f resources.json ]]; then mv resources.json "step${step}_resources.json"; fi
  if ((status!=0)); then exit "$status"; fi
done'
