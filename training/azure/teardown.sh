#!/usr/bin/env bash
# Stop paying for the GPU box. Two modes:
#
#   ./teardown.sh stop     deallocate the VM  (keeps disk + dataset; ~$10/mo)
#   ./teardown.sh delete   delete the whole resource group (nothing left)
#
# `stop` is the one to use between runs. `delete` also destroys the uploaded
# dataset bundle and any adapter you have not fetched yet.
set -euo pipefail

MODE=${1:-}
RG=${RG:-pdfbm-train}
VM=${VM:-pdfbm-gpu}

case "$MODE" in
stop)
    az vm deallocate -g "$RG" -n "$VM" -o none
    echo "deallocated $VM — compute billing stopped, disk still costs ~\$10/mo"
    ;;
delete)
    echo "This deletes resource group '$RG' and everything in it:"
    az resource list -g "$RG" -o tsv --query "[].[type,name]" || true
    printf "\nType the resource group name to confirm: "
    read -r reply
    if [ "$reply" != "$RG" ]; then
        echo "aborted" >&2
        exit 1
    fi
    az group delete -n "$RG" --yes --no-wait
    echo "delete requested for $RG (runs in the background)"
    ;;
*)
    sed -n '2,8p' "$0"
    exit 1
    ;;
esac
