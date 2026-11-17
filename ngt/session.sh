#!/usr/bin/env bash
set -euo pipefail
ngt_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PATH="$ngt_dir/.bin:$PATH"
if [[ ! -f "$ngt_dir/.env" ]]; then
    echo "Copy ngt/.env.example to ngt/.env and set CERN_USER first." >&2
    exit 1
fi
source "$ngt_dir/.env"
: "${CERN_USER:?Set CERN_USER in ngt/.env}"
: "${NGT_POD:=sdpa-w7900}"
: "${KUBECTL:=kubectl}"
if [[ ! "$NGT_POD" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ || ${#NGT_POD} -gt 63 ]]; then
    echo "NGT_POD must be a Kubernetes DNS label (at most 63 characters)." >&2
    exit 1
fi
export KUBECONFIG="$ngt_dir/.kube/config"
k=("$KUBECTL" --context ngt-prod --namespace "$CERN_USER")
case "${1:-help}" in
    init)
        mkdir -p "$ngt_dir/.kube"
        chmod 700 "$ngt_dir/.kube"
        if [[ ! -f "$KUBECONFIG" ]]; then
            (umask 077; curl --fail --location https://ngt-public.web.cern.ch/config -o "$KUBECONFIG")
        fi
        "${k[@]}" auth can-i list pods
        ;;
    create)
        # Keep the checked-in manifest usable directly; customize only its name.
        sed "s/name: sdpa-w7900$/name: $NGT_POD/" "$ngt_dir/session-w7900.yaml" | "${k[@]}" create -f -
        ;;
    status) "${k[@]}" get pod "$NGT_POD" -o wide ;;
    shell) "${k[@]}" exec -it "$NGT_POD" -- bash -l ;;
    copy)
        "${k[@]}" exec "$NGT_POD" -- mkdir -p /tmp/particleflow/scripts /tmp/particleflow/ngt
        "${k[@]}" cp "$ngt_dir/../scripts/benchmark-sdpa-onnx.py" "$NGT_POD:/tmp/particleflow/scripts/benchmark-sdpa-onnx.py"
        "${k[@]}" cp "$ngt_dir/setup-w7900.sh" "$NGT_POD:/tmp/particleflow/ngt/setup-w7900.sh"
        ;;
    fetch)
        mkdir -p "$ngt_dir/results"
        "${k[@]}" cp "$NGT_POD:/tmp/particleflow/results/." "$ngt_dir/results/"
        ;;
    delete) "${k[@]}" delete pod "$NGT_POD" ;;
    *) echo "Usage: bash ngt/session.sh {init|create|status|copy|shell|fetch|delete}" ;;
esac
