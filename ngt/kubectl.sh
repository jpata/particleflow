#!/usr/bin/env bash
set -euo pipefail
ngt_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ngt_dir/.env"
export PATH="$ngt_dir/.bin:$PATH"
export KUBECONFIG="$ngt_dir/.kube/config"
exec "${KUBECTL:-kubectl}" --context ngt-prod --namespace "${CERN_USER:?}" "$@"
