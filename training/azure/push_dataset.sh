#!/usr/bin/env bash
# Upload the dataset bundle built on the local machine to blob storage, where
# run_training.sh picks it up. See training/AZURE.md step 3.
set -euo pipefail

BUNDLE=${1:-dataset-bundle.zip}
RG=${RG:-pdfbm-train}
CONTAINER=${CONTAINER:-artifacts}
SUB=$(az account show --query id -o tsv)
STORAGE=${STORAGE:-pdfbm$(printf '%s' "$SUB" | shasum | cut -c1-12)}

if [ ! -f "$BUNDLE" ]; then
    echo "no such bundle: $BUNDLE" >&2
    echo "build it first: python training/build_dataset.py records.jsonl -o dataset/ && zip -r dataset-bundle.zip dataset/" >&2
    exit 1
fi

KEY=$(az storage account keys list -g "$RG" -n "$STORAGE" --query "[0].value" -o tsv)
az storage blob upload \
    --account-name "$STORAGE" --account-key "$KEY" \
    -c "$CONTAINER" -n dataset-bundle.zip -f "$BUNDLE" --overwrite -o none

printf 'uploaded %s (%s) to %s/%s\n' \
    "$BUNDLE" "$(du -h "$BUNDLE" | cut -f1)" "$STORAGE" "$CONTAINER"
