#!/usr/bin/env bash
# Run workflow steps inside the initialized CMSSW environment.
set -euo pipefail
run_dir=$PWD
first_step=${1:-3}
[[ $first_step =~ ^[1-5]$ ]] || { echo 'First step must be 1–5.' >&2; exit 2; }
if [[ $first_step == 3 && ! -f step2.root ]]; then
    echo 'Run from a directory containing step2.root.' >&2
    exit 2
fi
if [[ -f resources.json && ! -e step2_resources.json ]]; then
    mv resources.json step2_resources.json
fi
result=0
for ((step=first_step; step<=5; step++)); do
    if [[ $step == 1 ]]; then
        configs=("$run_dir/SinglePiFlatPt0p7To10_cfi_GEN_SIM.py")
    else
        configs=("$run_dir/step${step}_"*.py)
    fi
    [[ ${#configs[@]} == 1 && -f ${configs[0]} ]] || { echo "Expected one step${step} config" >&2; exit 2; }
    printf 'Running step%s: %s\n' "$step" "${configs[0]}"
    printf '%q ' cmsRun -j "$run_dir/step${step}_FrameworkJobReport.xml" "${configs[0]}" > "step${step}_command.txt"
    printf '\n' >> "step${step}_command.txt"
    set +e
    cmsRun -j "$run_dir/step${step}_FrameworkJobReport.xml" "${configs[0]}" 2>&1 | tee "step${step}_run.log"
    status=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$status" > "step${step}_exit_code"
    if [[ -f resources.json ]]; then mv resources.json "step${step}_resources.json"; fi
    printf 'step%s exit status: %s\n' "$step" "$status"
    if ((status != 0)); then
        result=1
        # Both later steps require step3 outputs. Step5 does not require step4.
        if ((step <= 3)); then exit "$status"; fi
    fi
done
exit "$result"
