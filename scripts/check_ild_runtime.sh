#!/usr/bin/env bash
# Check the native runtime on the submission host before starting a campaign.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE="$REPO_ROOT/mlpf/data/key4hep/gen/idea/idea-dev.sif"
test -r "$IMAGE"
apptainer exec --bind "/cvmfs,$REPO_ROOT,/tmp" \
    --env LD_PRELOAD=/usr/lib64/libffi.so.8 --env IDEA_STACK_RELEASE=Mon \
    "$IMAGE" bash -s -- "$REPO_ROOT" <<'CHECK'
set -euo pipefail
repo=$1
config="$repo/mlpf/data/key4hep/gen/idea"
source "$config/setup.sh"
assets=$K4GEO
idea_use_local "$config/k4geo-local"
idea_use_local "$config/k4RecTracker-local"
test "$(git -C "$repo/mlpf/data/key4hep/gen/ild/ILDConfig" rev-parse HEAD)" = fc1c9f58f9912b4d1fb10da5defbaca2133d60a0
command -v ddsim
command -v k4run
python -c 'import DDG4; from Configurables import TrackMerger; print("DDG4 and TrackMerger imports passed")'
check_dir=$(mktemp -d /tmp/ild-runtime-check.XXXXXXXX)
trap 'rm -rf -- "$check_dir"' EXIT
python "$repo/mlpf/data/key4hep/gen/ild/stage_geometry.py" "$K4GEO" "$check_dir/geometry" --assets "$assets"
printf 'ILD runtime check passed; stack checksum: %s\n' "$IDEA_STACK_ID"
CHECK
