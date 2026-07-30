#!/usr/bin/env bash
# Provision the Azure GPU box for the 8k/3B retrain. See training/AZURE.md.
#
# Idempotent: re-running skips anything that already exists, so it is safe to
# use as a "make sure the box is there" command. Creates nothing that bills
# more than a disk until the VM is running.
set -euo pipefail

RG=${RG:-pdfbm-train}
# swedencentral, not a region you would pick for latency: this subscription's
# Azure-side placement policy rejects every other region tried (see AZURE.md
# step 0). Change it only after probing that the target region is allowed.
LOC=${LOC:-swedencentral}
VM=${VM:-pdfbm-gpu}
SIZE=${SIZE:-Standard_NC4as_T4_v3}
CONTAINER=${CONTAINER:-artifacts}
OS_DISK_GB=${OS_DISK_GB:-128}
ADMIN=${ADMIN:-azureuser}
# Ubuntu 22.04 gen2: what the HpcCompute NVIDIA driver extension supports on
# NCasT4_v3. Torch wheels ship their own CUDA runtime, so only the driver.
IMAGE=${IMAGE:-Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest}

SUB=$(az account show --query id -o tsv)
# Storage account names are globally unique, 3-24 chars, lowercase alnum.
STORAGE=${STORAGE:-pdfbm$(printf '%s' "$SUB" | shasum | cut -c1-12)}

say() { printf '\n== %s\n' "$1"; }

say "subscription"
az account show --query "{name:name, id:id}" -o tsv

say "resource group $RG ($LOC)"
az group create -n "$RG" -l "$LOC" -o none

say "storage account $STORAGE"
if ! az storage account show -g "$RG" -n "$STORAGE" -o none 2>/dev/null; then
    az storage account create -g "$RG" -n "$STORAGE" -l "$LOC" \
        --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 -o none
fi
KEY=$(az storage account keys list -g "$RG" -n "$STORAGE" --query "[0].value" -o tsv)
az storage container create -n "$CONTAINER" --account-name "$STORAGE" \
    --account-key "$KEY" -o none

say "vm $VM ($SIZE)"
if az vm show -g "$RG" -n "$VM" -o none 2>/dev/null; then
    echo "already exists — skipping create"
else
    az vm create -g "$RG" -n "$VM" \
        --image "$IMAGE" \
        --size "$SIZE" \
        --admin-username "$ADMIN" \
        --generate-ssh-keys \
        --os-disk-size-gb "$OS_DISK_GB" \
        --storage-sku StandardSSD_LRS \
        --assign-identity \
        --custom-data "$(dirname "$0")/cloud-init.yaml" \
        --nsg-rule SSH \
        -o none
fi

# az vm create opens SSH to the whole internet; narrow it to this machine.
say "restricting ssh to this machine"
MYIP=$(curl -fsS https://ifconfig.me 2>/dev/null || true)
if [ -n "$MYIP" ]; then
    az network nsg rule update -g "$RG" --nsg-name "${VM}NSG" \
        -n default-allow-ssh --source-address-prefixes "$MYIP" -o none
    echo "ssh restricted to $MYIP (re-run this script if your IP changes)"
else
    echo "WARNING: could not determine public IP; SSH is open to 0.0.0.0/0" >&2
fi

say "nvidia driver extension (slow on first install, ~5 min)"
# Unpinned on purpose: the publisher ships 22.04 support from 1.9 onward and
# the newest (1.13.x today) is what we want. Pin --version only to reproduce.
az vm extension set -g "$RG" --vm-name "$VM" \
    -n NvidiaGpuDriverLinux --publisher Microsoft.HpcCompute \
    -o none

# The VM zips the adapter to blob storage and then deallocates itself at the
# end of the run, so a finished job stops billing without us babysitting it.
say "granting the vm identity blob write + self-deallocate"
PRINCIPAL=$(az vm show -g "$RG" -n "$VM" --query identity.principalId -o tsv)
VM_ID=$(az vm show -g "$RG" -n "$VM" --query id -o tsv)
STORAGE_ID=$(az storage account show -g "$RG" -n "$STORAGE" --query id -o tsv)
for pair in "Storage Blob Data Contributor|$STORAGE_ID" "Virtual Machine Contributor|$VM_ID"; do
    role=${pair%%|*}; scope=${pair##*|}
    az role assignment create --assignee-object-id "$PRINCIPAL" \
        --assignee-principal-type ServicePrincipal \
        --role "$role" --scope "$scope" -o none 2>/dev/null \
        || echo "role '$role' already assigned"
done

IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
cat <<EOF

== ready
  vm       $VM  ($SIZE, \$0.558/hr in $LOC — deallocate when idle)
  ssh      ssh $ADMIN@$IP
  storage  $STORAGE / $CONTAINER

next (training/AZURE.md steps 3-5):
  ./training/azure/push_dataset.sh dataset-bundle.zip
  ssh $ADMIN@$IP 'bash -s' < training/azure/run_training.sh   # or run it on the box
  ./training/azure/fetch_adapter.sh
EOF
