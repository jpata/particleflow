#!/usr/bin/env bash
# Check the shared native runtime on the submission host before a campaign.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GENERATOR_ROOT="$REPO_ROOT/mlpf/data/key4hep/gen"
IMAGE="$GENERATOR_ROOT/idea/idea-dev.sif"
test -r "$IMAGE"
apptainer exec --bind "/cvmfs,$REPO_ROOT,/tmp" \
    --env LD_PRELOAD=/usr/lib64/libffi.so.8 --env KEY4HEP_RELEASE=2026-04-08 \
    "$IMAGE" bash "$GENERATOR_ROOT/check_runtime.sh"
